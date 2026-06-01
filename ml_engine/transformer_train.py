import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import numpy as np
import joblib
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from ml_engine.train import prepare_training_data
from ml_engine.sequence_dataset import build_sequence_dataset
from ml_engine.transformer_model import PitchTransformer
from ml_engine.config import ALLOWED_FEATURES

MODEL_DIR   = Path('ml_engine/models')
BATCH_SIZE  = 512
EPOCHS      = 50
PATIENCE    = 10
LR          = 1e-3

# ID 계열 피처 및 시퀀스 생성용 이전 구종 제외 피처 목록
ID_FEATURES = ['pitcher', 'batter', 'fielder_2', 'game_year',
               'prev_pitch_1', 'prev_pitch_2', 'prev_pitch_3']

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

def train_transformer(sampling_rate=0.1):
    device = get_device()
    print(f"[디바이스] {device}")

    print("[*] 데이터 로드 중...")
    _, _, _, _, feat_names, label_encoder, df = \
        prepare_training_data(sampling_rate=sampling_rate, return_df=True)

    n_classes = len(label_encoder.classes_)

    # 1. 스케일링 대상 피처 선정 (ID 계열 제외)
    scale_features = [
        f for f in ALLOWED_FEATURES 
        if f in df.columns and f not in ID_FEATURES
    ]

    # 2. StandardScaler 학습 및 적용 (2024년 시즌 기준으로 fit)
    scaler = StandardScaler()
    train_mask = df['game_year'] == 2024
    
    scaler.fit(df.loc[train_mask, scale_features])
    df[scale_features] = scaler.transform(df[scale_features])
    
    print(f"[스케일링] 대상 피처 수: {len(scale_features)}개")

    # 3. NN 입력 피처 선정 (학습 불필요한 ID 계열 제외)
    nn_features = [
        f for f in ALLOWED_FEATURES
        if f not in ['pitcher', 'batter', 'fielder_2', 'game_year']
    ]
    feature_dim = len(nn_features)
    print(f"[NN 입력 피처] {feature_dim}개 (ID 계열 제외)")

    # 4. 3D 시퀀스 텐서 데이터셋 구축
    print("[*] 3D 시퀀스 데이터셋 빌드 중...")
    train_dataset = build_sequence_dataset(df, label_encoder, split='train', features=nn_features)
    val_dataset = build_sequence_dataset(df, label_encoder, split='test', features=nn_features)

    # 5. [Batch, 5, F] 텐서 shape 검증 출력
    print(f"\n[*] [Batch, 5, F] 텐서 shape 검증:")
    print(f"    - Train X shape: {train_dataset.X.shape}")
    print(f"    - Val X shape: {val_dataset.X.shape}")
    print(f"    - Feature_dim (F): {feature_dim}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = PitchTransformer(feature_dim=feature_dim, n_classes=n_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] 파라미터 수: {total_params:,}")

    # 클래스 불균형 완화 가중치 계산
    y_train = train_dataset.y.numpy()
    class_counts = np.bincount(y_train)
    weights = 1.0 / np.sqrt(class_counts + 1e-6)
    weights = weights / weights.sum() * n_classes
    class_weights = torch.FloatTensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer, max_lr=LR,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS, pct_start=0.1
    )

    best_f1, patience_cnt = 0.0, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        model.eval()
        preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds.append(model(xb).argmax(1).cpu().numpy())
                all_labels.append(yb.numpy())
        preds  = np.concatenate(preds)
        all_labels = np.concatenate(all_labels)
        val_f1 = f1_score(all_labels, preds, average='weighted', zero_division=0)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {total_loss/steps_per_epoch:.4f} | Val F1: {val_f1*100:.2f}%")

        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_cnt = 0
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / 'transformer_pitch_model.pt')
            joblib.dump(scaler,          MODEL_DIR / 'transformer_scaler.pkl',          compress=3)
            joblib.dump(scale_features,  MODEL_DIR / 'transformer_scale_features.pkl',  compress=3)
            joblib.dump(nn_features,     MODEL_DIR / 'transformer_nn_features.pkl',     compress=3)
            print(f"  → Best 저장 (F1: {best_f1*100:.2f}%)")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"[EarlyStopping] {PATIENCE} epoch 개선 없음.")
                break

    print(f"\n[완료] Best Transformer F1: {best_f1*100:.2f}%")
    print(f"[비교] XGBoost: 42.52% / Transformer (5구 시퀀스): {best_f1*100:.2f}%")
    return best_f1

if __name__ == '__main__':
    train_transformer(sampling_rate=0.1)
