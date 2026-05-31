import sys
import gc
import torch
import numpy as np
import joblib
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from ml_engine.train import prepare_training_data
from ml_engine.sequence_dataset import build_sequence_dataset
from ml_engine.transformer_model import PitchTransformer

MODEL_DIR = Path('ml_engine/models')

def evaluate_transformer():
    print("[*] Transformer 100% 검증 데이터 정밀 평가 진입")
    sys.stdout.flush()

    try:
        # 1. 100% 전체 데이터 로드 및 label encoder 로딩
        _, _, _, _, feat_names, label_encoder, df = prepare_training_data(sampling_rate=1.0, return_df=True)
        n_classes = len(label_encoder.classes_)
        print(f"  [확인] 전체 클래스 수: {n_classes}개")
        sys.stdout.flush()

        # 2. 저장된 Transformer 아티팩트 로드
        scaler = joblib.load(MODEL_DIR / 'transformer_scaler.pkl')
        scale_features = joblib.load(MODEL_DIR / 'transformer_scale_features.pkl')
        nn_features = joblib.load(MODEL_DIR / 'transformer_nn_features.pkl')
        feature_dim = len(nn_features)

        # 3. StandardScaler 적용
        df[scale_features] = scaler.transform(df[scale_features])

        # 4. 데이터셋 빌드
        val_dataset = build_sequence_dataset(df, label_encoder, split='test', features=nn_features)
        val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

        # 5. 모델 빌드 및 로딩
        model = PitchTransformer(feature_dim=feature_dim, n_classes=n_classes)
        
        # CPU로 state_dict 로드
        state_dict = torch.load(MODEL_DIR / 'transformer_pitch_model.pt', map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()

        # 6. 추론 및 F1 Score 측정
        preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds.append(model(xb).argmax(1).numpy())
                all_labels.append(yb.numpy())

        preds = np.concatenate(preds)
        all_labels = np.concatenate(all_labels)

        val_f1 = f1_score(all_labels, preds, average='weighted', zero_division=0)
        print(f"\n[평가 완료]")
        print(f"  🏆 Transformer 2025 검증 Weighted F1-Score: {val_f1 * 100:.2f}%")
        sys.stdout.flush()

    except Exception as e:
        import traceback
        print("\n[에러 발생]")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        raise e

if __name__ == "__main__":
    evaluate_transformer()
