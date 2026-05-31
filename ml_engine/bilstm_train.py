import torch
import torch.nn as nn
import numpy as np
import joblib
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

from ml_engine.train import prepare_training_data
from ml_engine.sequence_dataset import build_sequence_dataset
from ml_engine.bilstm_model import PitchBiLSTM
from ml_engine.config import ALLOWED_FEATURES

MODEL_DIR = Path('ml_engine/models')
BATCH_SIZE = 256
EPOCHS = 50
PATIENCE = 15
LR = 3e-4

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def train_bilstm(sampling_rate: float = 0.1):
    device = get_device()
    print(f"[디바이스] {device}")

    # - 데이터 준비 (XGBoost 파이프라인 재사용 + return_df=True 수집)
    print("[*] 데이터 로드 중...")
    X_train_raw, X_test_raw, y_train_raw, y_test_raw, feat_names, label_encoder, df = \
        prepare_training_data(sampling_rate=sampling_rate, return_df=True)

    n_classes = len(label_encoder.classes_)

    # ------------------------------------------------------------------
    # 단계 A-1. 피처 스케일링 (StandardScaler) 적용
    # ------------------------------------------------------------------
    from sklearn.preprocessing import StandardScaler

    # - ID 계열 피처 (스케일링 제외 대상)
    ID_FEATURES = ['pitcher', 'batter', 'fielder_2', 'game_year',
                   'prev_pitch_1', 'prev_pitch_2', 'prev_pitch_3']

    # - 스케일링 대상 피처 선정
    scale_features = [
        f for f in ALLOWED_FEATURES 
        if f in df.columns and f not in ID_FEATURES
    ]

    # - StandardScaler 학습 및 적용 (2024년 시즌 기준으로 fit)
    scaler = StandardScaler()
    train_mask = df['game_year'] == 2024

    scaler.fit(df.loc[train_mask, scale_features])
    df[scale_features] = scaler.transform(df[scale_features])

    # - 스케일러 저장 (inference 시 재사용)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, MODEL_DIR / 'bilstm_scaler.pkl', compress=3)
    joblib.dump(scale_features, MODEL_DIR / 'bilstm_scale_features.pkl', compress=3)
    print(f"[스케일링] 대상 피처 수: {len(scale_features)}개")

    # ------------------------------------------------------------------
    # 단계 A-2. 신경망 입력에서 ID 피처 제거
    # ------------------------------------------------------------------
    nn_features = [
        f for f in ALLOWED_FEATURES
        if f not in ['pitcher', 'batter', 'fielder_2', 'game_year']
    ]
    feature_dim = len(nn_features)
    print(f"[NN 입력 피처] {feature_dim}개 (ID 계열 제외)")

    # - 3D 시퀀스 텐서 데이터셋 구축
    print("[*] 3D 시퀀스 데이터셋 빌드 중...")
    train_dataset = build_sequence_dataset(df, label_encoder, split='train', features=nn_features)
    val_dataset = build_sequence_dataset(df, label_encoder, split='test', features=nn_features)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # - best_params 로드 후 hidden_size 절반 적용 (과적합 억제)
    import json
    params_path = MODEL_DIR / "best_params_bilstm.json"
    bp = json.load(open(params_path)) if params_path.exists() else {}
    hidden_size  = max(32, bp.get("hidden_size", 128) // 2)  # 128 → 64
    num_layers   = bp.get("num_layers", 2)
    dropout      = bp.get("dropout", 0.45)
    print(f"[BiLSTM 파라미터] hidden_size={hidden_size} (best 절반), num_layers={num_layers}, dropout={dropout:.3f}")

    # - 모델 초기화
    model = PitchBiLSTM(
        feature_dim=feature_dim,
        n_classes=n_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # - 클래스 불균형 완화 (스퀘어루트 기반 가중치로 안정성과 소수 클래스 탐색성 동시 확보)
    y_train = train_dataset.y.numpy()
    class_counts = np.bincount(y_train)
    weights = 1.0 / (np.sqrt(class_counts) + 1e-6)
    weights = weights / weights.sum() * n_classes
    class_weights = torch.FloatTensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=1e-3  # 1e-4 → 1e-3 (과적합 억제)
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_f1 = 0.0
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        train_preds = []
        train_labels = []

        # 배치 단위 학습
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

            preds = logits.argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels.extend(yb.cpu().numpy())

        scheduler.step()

        # 검증
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        train_f1 = f1_score(train_labels, train_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {total_loss/len(train_dataset)*BATCH_SIZE:.4f} | Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_cnt = 0
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / 'bilstm_pitch_model.pt')
            print(f"  → Best 모델 저장 (F1: {best_f1:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"[EarlyStopping] {PATIENCE} epoch 개선 없음. 학습 종료.")
                break

    print(f"\n[완료] Best Bi-LSTM F1: {best_f1:.4f} ({best_f1*100:.2f}%)")
    joblib.dump(label_encoder, MODEL_DIR / 'label_encoder.pkl', compress=3)
    return best_f1


if __name__ == '__main__':
    # - 빠른 검증 및 차원 일치를 위해 10% 샘플링 비율로 고속 학습 실행
    train_bilstm(sampling_rate=0.1)
