import sys
import gc
import json
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import f1_score

from ml_engine.train import prepare_training_data

MODEL_DIR = Path('ml_engine/models')
BATCH_SIZE = 512

def batch_predict_proba(model, X, batch_size=100000):
    probas = []
    for i in range(0, len(X), batch_size):
        chunk = X[i:i+batch_size]
        probas.append(model.predict_proba(chunk))
    return np.vstack(probas)

def find_best_weights():
    print("[*] STACK (트리 앙상블) + Transformer 앙상블 최적화 진입 (메모리 격리 모드)")
    sys.stdout.flush()

    try:
        # 1. 100% 데이터 로딩 (2025년 전체 검증 데이터 기준 그리드 서치)
        print("[*] 데이터 로드 및 가공 중...")
        sys.stdout.flush()
        X_train_raw, X_test_raw, y_train_raw, y_test, feat_names, label_encoder_10pct, df = \
            prepare_training_data(sampling_rate=1.0, return_df=True)

        from ml_engine.config import TEST_YEAR
        print("  - [메모리 최적화] 2024년 학습 대용량 데이터 제거 및 RAM 확보 중...")
        sys.stdout.flush()
        df = df[df['game_year'] == TEST_YEAR].copy()
        
        # prepare_training_data 결과물에서 17-Class Label Encoder를 온전히 확보합니다.
        label_encoder = label_encoder_10pct
        n_classes = len(label_encoder.classes_)
        
        del X_train_raw, y_train_raw, label_encoder_10pct
        gc.collect()

        # 인코더에 없는 구종 필터링
        print("[*] 17-Class 인코더에 없는 구종 필터링 중...")
        sys.stdout.flush()
        valid_test_idx = df['pitch_type'].isin(label_encoder.classes_).values
        df = df[valid_test_idx].copy()
        X_test_raw = X_test_raw.iloc[valid_test_idx].copy()
        y_test_filtered = y_test[valid_test_idx]

        X_test_filled = X_test_raw.fillna(0).values.astype(np.float32)

        # ------------------------------------------------------------------
        # A. 순차 로딩 & 즉각 해제로 메모리 OOM 원천 방어
        # ------------------------------------------------------------------
        print("  - [순차 로드 1/3] XGBoost 추론 중...")
        sys.stdout.flush()
        xgb_model = joblib.load(MODEL_DIR / 'xgboost_pitch_model.pkl')
        xgb_probas = batch_predict_proba(xgb_model, X_test_filled)
        del xgb_model
        gc.collect()

        print("  - [순차 로드 2/3] LightGBM 추론 중...")
        sys.stdout.flush()
        lgb_model = joblib.load(MODEL_DIR / 'lightgbm_pitch_model.pkl')
        lgb_probas = batch_predict_proba(lgb_model, X_test_filled)
        del lgb_model
        gc.collect()

        print("  - [순차 로드 3/3] CatBoost 추론 중...")
        sys.stdout.flush()
        cat_model = joblib.load(MODEL_DIR / 'catboost_pitch_model.pkl')
        cat_probas = batch_predict_proba(cat_model, X_test_filled)
        del cat_model
        gc.collect()

        # 51차원 메타 피처 조립
        print("[*] 메타 피처 조립 중...")
        sys.stdout.flush()
        X_test_meta = np.hstack([xgb_probas, lgb_probas, cat_probas])
        del xgb_probas, lgb_probas, cat_probas, X_test_filled
        gc.collect()

        # 메타 러너 단독 로드 & 예측 후 해제
        print("[*] Stacking 메타러너 추론 중...")
        sys.stdout.flush()
        meta_learner = joblib.load(MODEL_DIR / 'stacking_meta_learner.pkl')
        stack_probas = batch_predict_proba(meta_learner, X_test_meta)
        del meta_learner, X_test_meta
        gc.collect()

        # ------------------------------------------------------------------
        # B. Transformer 예측 (CPU 강제 고정으로 MPS 드라이버 충돌 방지)
        # ------------------------------------------------------------------
        print("[*] PyTorch 및 Transformer 종속성 지연 로딩 중...")
        sys.stdout.flush()
        import torch
        torch.set_num_threads(1)
        from torch.utils.data import DataLoader
        from ml_engine.sequence_dataset import build_sequence_dataset
        from ml_engine.transformer_model import PitchTransformer

        print("[*] Transformer 최적 HPO 설정 로드 중...")
        sys.stdout.flush()
        with open(MODEL_DIR / 'best_params_transformer.json', 'r') as f:
            t_params = json.load(f)
            
        seq_len = t_params.get('sequence_length', 10)
        d_model = t_params['d_model']
        nhead = t_params['nhead']
        num_layers = t_params['num_layers']
        dim_feedforward = t_params['dim_feedforward']
        dropout = t_params['dropout']

        print(f"  [설정] sequence_length={seq_len}, d_model={d_model}, nhead={nhead}")
        sys.stdout.flush()

        scaler = joblib.load(MODEL_DIR / 'transformer_scaler.pkl')
        scale_features = joblib.load(MODEL_DIR / 'transformer_scale_features.pkl')
        nn_features = joblib.load(MODEL_DIR / 'transformer_nn_features.pkl')
        feature_dim = len(nn_features)

        df_copy = df.copy()
        df_copy[scale_features] = scaler.transform(df_copy[scale_features])

        print("[*] Transformer 3D 데이터셋 빌드 중...")
        sys.stdout.flush()
        val_dataset = build_sequence_dataset(df_copy, label_encoder, split='test', features=nn_features, seq_len=seq_len)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # CPU 추론 강제 고정
        device = torch.device('cpu')
        model = PitchTransformer(
            feature_dim=feature_dim, n_classes=n_classes,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            dim_feedforward=dim_feedforward, dropout=dropout
        ).to(device)
        
        model.load_state_dict(torch.load(MODEL_DIR / 'transformer_pitch_model.pt', map_location='cpu'))
        model.eval()

        print("[*] Transformer CPU 추론 실행 중...")
        sys.stdout.flush()
        transformer_probas_list = []
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                probas = torch.softmax(model(xb), dim=1).numpy()
                transformer_probas_list.append(probas)

        transformer_probas = np.vstack(transformer_probas_list)
        del transformer_probas_list, model
        gc.collect()

        # 시퀀스 데이터셋 기준 라벨
        y_test_seq = val_dataset.y.numpy()

        # ------------------------------------------------------------------
        # C. 형상 일치 보정
        # ------------------------------------------------------------------
        n_min = min(len(stack_probas), len(transformer_probas), len(y_test_seq))
        if n_min < max(len(stack_probas), len(transformer_probas), len(y_test_seq)):
            print(f"  ⚠️ 형상 불일치 감지 -> 공통 {n_min}행으로 보정")
            stack_probas = stack_probas[:n_min]
            transformer_probas = transformer_probas[:n_min]
            y_test_seq = y_test_seq[:n_min]

        # ------------------------------------------------------------------
        # D. 단독 모델 F1 측정
        # ------------------------------------------------------------------
        stack_preds = stack_probas.argmax(axis=1)
        transformer_preds = transformer_probas.argmax(axis=1)

        stack_f1 = f1_score(y_test_seq, stack_preds, average='weighted', zero_division=0)
        transformer_f1 = f1_score(y_test_seq, transformer_preds, average='weighted', zero_division=0)

        print(f"\n[단독 모델 성능 (100% 전체 데이터 기준)]")
        print(f"  XGBoost/Stacking 단독 F1: {stack_f1 * 100:.2f}%")
        print(f"  Transformer 단독 F1:      {transformer_f1 * 100:.2f}%")
        sys.stdout.flush()

        # ------------------------------------------------------------------
        # E. 그리드 탐색 블렌딩 (0.05 단위 w 탐색)
        # ------------------------------------------------------------------
        best_f1, best_w = 0.0, 1.0
        print("\n[STACK + Transformer 블렌딩 가중치 튜닝 실험 시작]")
        sys.stdout.flush()
        
        for w in np.arange(0.0, 1.01, 0.05):
            blended = w * stack_probas + (1.0 - w) * transformer_probas
            preds = blended.argmax(axis=1)
            f1 = f1_score(y_test_seq, preds, average='weighted', zero_division=0)
            print(f"stack_w={w:.2f}, transformer_w={1.0 - w:.2f} → F1={f1:.4f}")
            sys.stdout.flush()
            
            if f1 > best_f1:
                best_f1 = f1
                best_w = w

        # ------------------------------------------------------------------
        # F. 최적화 피클 파일 저장 (xgb_w에 stack_w 배정, lstm_w에 transformer_w 배정)
        # ------------------------------------------------------------------
        weights = {'xgb_w': float(best_w), 'lstm_w': float(1.0 - best_w)}
        joblib.dump(weights, MODEL_DIR / 'ensemble_weights.pkl', compress=3)

        print(f"\n[최적화 완료]")
        print(f"  🏆 최적 앙상블: STACK {best_w:.2f} / Transformer {1.0 - best_w:.2f}")
        print(f"  🥇 최고 앙상블 F1-Score: {best_f1 * 100:.2f}%")
        print(f"  💾 ensemble_weights.pkl 안전하게 저장 완료")
        sys.stdout.flush()

    except Exception as e:
        import traceback
        print("\n[에러 발생]")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise e

if __name__ == "__main__":
    find_best_weights()
