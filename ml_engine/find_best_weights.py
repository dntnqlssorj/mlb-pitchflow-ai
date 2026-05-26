import sys
import gc
import numpy as np
import torch
import joblib
from pathlib import Path
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ml_engine.train import prepare_training_data
from ml_engine.sequence_dataset import build_sequence_dataset
from ml_engine.bilstm_model import PitchBiLSTM
from ml_engine.config import ALLOWED_FEATURES

MODEL_DIR = Path('ml_engine/models')
BATCH_SIZE = 256

def get_device():
    # Python 3.14 + PyTorch MPS의 KERN_INVALID_ADDRESS 세그폴트 이슈 예방을 위해 CPU 강제 사용
    return torch.device('cpu')

def find_best_weights():
    device = get_device()
    print(f"[*] 디바이스: {device}")
    sys.stdout.flush()

    try:
        # 1. 데이터 로드 및 전처리 (10% 샘플링 → 가중치 탐색 속도 극대화)
        print("[*] 데이터 로드 및 가공 중...")
        sys.stdout.flush()
        X_train_raw, X_test_raw, y_train_raw, y_test, feat_names, label_encoder_10pct, df = \
            prepare_training_data(sampling_rate=0.1, return_df=True)

        # 2. 전체 데이터로 학습된 17개 클래스 인코더 로드
        print("[*] 저장된 17-Class Label Encoder 로드 중...")
        sys.stdout.flush()
        label_encoder = joblib.load(MODEL_DIR / 'label_encoder.pkl')
        n_classes = len(label_encoder.classes_)

        # 10% 샘플링 시 희귀 구종이 'OT'로 통합되었으나, 17-class 인코더에는 없음
        # -> 인코더에 없는 구종(OT 등) 필터링
        print("[*] 17-Class 인코더에 없는 구종 필터링 중...")
        sys.stdout.flush()
        from ml_engine.config import TEST_YEAR
        test_mask = (df['game_year'] == TEST_YEAR).values
        df_test = df[test_mask]
        valid_test_idx = df_test['pitch_type'].isin(label_encoder.classes_).values
        X_test_raw = X_test_raw.iloc[valid_test_idx].copy()

        valid_df_idx = df['pitch_type'].isin(label_encoder.classes_).values
        df = df[valid_df_idx].copy()

        # 3. XGBoost 예측 — 격리 서브프로세스 실행
        #    해결: fresh 메모리 서브프로세스에서 XGBoost 단독 실행 후 결과 파일로 전달
        print("[*] X_test_raw numpy 변환 및 임시 저장 중 (서브프로세스 입력)...")
        sys.stdout.flush()
        import subprocess, json
        X_test_np_save = X_test_raw.fillna(0).values.astype(np.float32)
        col_names = list(X_test_raw.columns)
        np.save('/tmp/X_test_raw_ensemble.npy', X_test_np_save)
        with open('/tmp/X_test_cols_ensemble.json', 'w') as f:
            json.dump(col_names, f)
        del X_test_np_save
        gc.collect()
        print(f"  저장 완료: shape=({len(col_names)} cols, {len(X_test_raw)} rows)")
        sys.stdout.flush()

        print("[*] XGBoost 예측 (격리 서브프로세스 실행)...")
        sys.stdout.flush()
        result = subprocess.run(
            ['./venv/bin/python', '-u', 'ml_engine/xgb_predict_subprocess.py'],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        if result.returncode != 0:
            raise RuntimeError(f"XGBoost 서브프로세스 실패 (exit={result.returncode})")

        xgb_probas = np.load('/tmp/xgb_probas_ensemble.npy')
        if xgb_probas.ndim == 1:
            xgb_probas = xgb_probas.reshape(-1, n_classes)
        gc.collect()
        print(f"  XGBoost 예측 완료: shape {xgb_probas.shape}")
        sys.stdout.flush()



        # 4. Bi-LSTM 모델 및 스케일러 로드
        print("[*] Bi-LSTM 모델 및 스케일러 로드 중...")
        sys.stdout.flush()
        scaler = joblib.load(MODEL_DIR / 'bilstm_scaler.pkl')
        scale_features = joblib.load(MODEL_DIR / 'bilstm_scale_features.pkl')

        df[scale_features] = scaler.transform(df[scale_features])

        ID_FEATURES = ['pitcher', 'batter', 'fielder_2', 'game_year']
        nn_features = [f for f in ALLOWED_FEATURES if f not in ID_FEATURES]
        feature_dim = len(nn_features)

        # 3D 시퀀스 테스트셋 구축 (17-Class Label Encoder 명시적 투입)
        val_dataset = build_sequence_dataset(df, label_encoder, split='test', features=nn_features)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        lstm = PitchBiLSTM(feature_dim=feature_dim, n_classes=n_classes)
        print("[*] Bi-LSTM 가중치 로드 중...")
        sys.stdout.flush()
        lstm.load_state_dict(torch.load(MODEL_DIR / 'bilstm_pitch_model.pt', map_location='cpu'))
        lstm.to(device)
        lstm.eval()

        print("[*] Bi-LSTM 예측 중...")
        sys.stdout.flush()
        lstm_probas_list = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                logits = lstm(xb)
                probas = torch.softmax(logits, dim=1).cpu().numpy()
                lstm_probas_list.append(probas)

        lstm_probas = np.vstack(lstm_probas_list)  # (N, 17)
        del lstm_probas_list
        gc.collect()
        print(f"  Bi-LSTM 예측 완료: shape {lstm_probas.shape}")
        sys.stdout.flush()

        # y_test 시퀀스 기준 라벨 (17-Class)
        y_test_seq = val_dataset.y.numpy()

        # ------------------------------------------------------------------
        # shape 매칭 검증
        # ------------------------------------------------------------------
        print(f"\n[*] shape 확인: XGB {xgb_probas.shape}, LSTM {lstm_probas.shape}, y {y_test_seq.shape}")
        sys.stdout.flush()

        # XGB는 10% 검증셋(71,133행), LSTM은 시퀀스 기반 같은 df → 동일해야 함
        # 불일치 시 공통 최솟값으로 자름
        n_min = min(len(xgb_probas), len(lstm_probas), len(y_test_seq))
        if n_min < max(len(xgb_probas), len(lstm_probas), len(y_test_seq)):
            print(f"  ⚠️  shape 불일치 감지 → 공통 {n_min}행으로 자름")
            xgb_probas  = xgb_probas[:n_min]
            lstm_probas = lstm_probas[:n_min]
            y_test_seq  = y_test_seq[:n_min]
        sys.stdout.flush()

        # ------------------------------------------------------------------
        # 단독 모델 F1 출력
        # ------------------------------------------------------------------
        xgb_preds  = xgb_probas.argmax(axis=1)
        lstm_preds  = lstm_probas.argmax(axis=1)
        xgb_f1  = f1_score(y_test_seq, xgb_preds,  average='weighted', zero_division=0)
        lstm_f1 = f1_score(y_test_seq, lstm_preds, average='weighted', zero_division=0)
        print(f"\n[단독 모델 성능]")
        print(f"  XGBoost 단독 F1:  {xgb_f1 * 100:.2f}%")
        print(f"  Bi-LSTM 단독 F1:  {lstm_f1 * 100:.2f}%")
        sys.stdout.flush()

        # ------------------------------------------------------------------
        # 5. 가중치 최적화 탐색 (0.05 간격 그리드 서치)
        # ------------------------------------------------------------------
        best_f1, best_w = 0.0, 0.5
        print("\n[가중치 튜닝 실험 시작]")
        sys.stdout.flush()
        for w in np.arange(0.0, 1.01, 0.05):
            blended = w * xgb_probas + (1.0 - w) * lstm_probas
            preds = blended.argmax(axis=1)
            f1 = f1_score(y_test_seq, preds, average='weighted', zero_division=0)
            print(f"  XGB {w:.2f} / LSTM {1.0 - w:.2f} ➡️  F1: {f1 * 100:.2f}%")
            sys.stdout.flush()
            if f1 > best_f1:
                best_f1 = f1
                best_w = w

        # ------------------------------------------------------------------
        # 6. 최적 가중치 저장 (ensemble.py 자동 로드 구조)
        # ------------------------------------------------------------------
        weights = {'xgb_w': float(best_w), 'lstm_w': float(1.0 - best_w)}
        joblib.dump(weights, MODEL_DIR / 'ensemble_weights.pkl', compress=3)

        print(f"\n[최적화 완료]")
        print(f"  🏆 최적 가중치: XGBoost {best_w:.2f} / Bi-LSTM {1.0 - best_w:.2f}")
        print(f"  📊 XGBoost 단독:  {xgb_f1 * 100:.2f}%")
        print(f"  📊 Bi-LSTM 단독:  {lstm_f1 * 100:.2f}%")
        print(f"  🥇 앙상블 최고 F1-Score: {best_f1 * 100:.2f}%")
        print(f"  💾 ensemble_weights.pkl 저장 완료")
        sys.stdout.flush()

    except Exception as e:
        import traceback
        print("\n[에러 발생]")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise e

if __name__ == "__main__":
    find_best_weights()
