import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ml_engine.train import prepare_training_data

def train_catboost(sampling_rate: float = 1.0):
    """
    [CatBoost 학습 파이프라인]
    - prepare_training_data() 파이프라인 재사용
    - CatBoostClassifier를 사용하여 학습 및 평가 진행 후 모델 영구 저장
    """
    # ------------------------------------------------------------------
    # 단계 1. 데이터 로드 및 전처리 (Chronological Split)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, feat_names, label_encoder = prepare_training_data(
        sampling_rate=sampling_rate
    )
    
    # ------------------------------------------------------------------
    # 단계 2. CatBoost 모델 학습
    # ------------------------------------------------------------------
    print("\nCatBoost 모델 학습 시작...")
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.1,
        loss_function='MultiClass',
        random_seed=42,
        verbose=100
    )
    
    model.fit(X_train, y_train)
    
    # ------------------------------------------------------------------
    # 단계 3. 모델 평가
    # ------------------------------------------------------------------
    print("\n[CatBoost 평가 중...]")
    y_pred = model.predict(X_test)
    if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.ravel()
        
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred,  average='weighted', zero_division=0)
    f1   = f1_score(y_test, y_pred,      average='weighted', zero_division=0)
    
    print("\n[CatBoost 성능 결과]")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # ------------------------------------------------------------------
    # 단계 4. 아티팩트 저장 및 패킹
    # ------------------------------------------------------------------
    model_dir = Path('ml_engine/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path   = model_dir / 'catboost_pitch_model.pkl'
    encoder_path = model_dir / 'label_encoder.pkl'
    
    joblib.dump(model, model_path, compress=3)
    joblib.dump(label_encoder, encoder_path, compress=3)
    
    print(f"\nCatBoost 모델 패킹 완료:")
    print(f"  모델 저장 경로:   {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    print(f"  인코더 저장 경로: {encoder_path} ({encoder_path.stat().st_size / 1024:.1f} KB)")
    
    return f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling', type=float, default=1.0, help='데이터 샘플링 비율 (기본값: 1.0)')
    args = parser.parse_args()
    
    train_catboost(sampling_rate=args.sampling)
