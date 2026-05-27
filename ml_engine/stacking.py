import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ml_engine.train import prepare_training_data

def train_stacking(sampling_rate: float = 1.0):
    """
    [OOF Stacking 앙상블 학습 파이프라인]
    - Level-0 기저 모델: XGBoost, LightGBM, CatBoost
    - OOF 교차검증 기반 메타 피처 (3 * N_CLASS) 행렬 빌드
    - Level-1 메타 러너: XGBoost (Max Depth 3) 학습 및 평가
    - 검증 및 아티팩트(meta_learner, base_model_paths) 패킹
    """
    # ------------------------------------------------------------------
    # 단계 1. 데이터 로드 및 분할 (prepare_training_data 재사용)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, feat_names, label_encoder = prepare_training_data(
        sampling_rate=sampling_rate
    )
    
    n_classes = len(label_encoder.classes_)
    n_samples = len(X_train)
    print(f"\n[Stacking OOF 피처 생성 시작] 학습 샘플 수: {n_samples:,}, 클래스 수: {n_classes}")
    
    # ------------------------------------------------------------------
    # 단계 2. OOF 교차 검증 및 메타 피처 생성
    # ------------------------------------------------------------------
    oof_xgb = np.zeros((n_samples, n_classes))
    oof_lgb = np.zeros((n_samples, n_classes))
    oof_cat = np.zeros((n_samples, n_classes))
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold+1} / 5 OOF 학습 진행 중 ---")
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        xgb_fold = XGBClassifier(
            n_estimators=541,
            max_depth=4,
            learning_rate=0.03716146208921116,
            subsample=0.6371106085974699,
            colsample_bytree=0.9923202402440235,
            min_child_weight=1,
            gamma=0.40225152117187357,
            reg_alpha=0.85331092958197,
            reg_lambda=1.0860868437421698,
            random_state=42,
            n_jobs=-1,
            eval_metric='mlogloss',
            enable_categorical=False
        )
        lgb_fold = LGBMClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, verbose=-1)
        cat_fold = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, loss_function='MultiClass', random_seed=42, verbose=0)
        
        xgb_fold.fit(X_tr, y_tr)
        lgb_fold.fit(X_tr, y_tr)
        cat_fold.fit(X_tr, y_tr)
        
        oof_xgb[val_idx] = xgb_fold.predict_proba(X_val)
        oof_lgb[val_idx] = lgb_fold.predict_proba(X_val)
        oof_cat[val_idx] = cat_fold.predict_proba(X_val)
        print(f"Fold {fold+1} 완료.")
        
    X_train_meta = np.hstack([oof_xgb, oof_lgb, oof_cat])
    print(f"\nOOF 메타 피처 결합 완료: Shape={X_train_meta.shape}")
    
    # ------------------------------------------------------------------
    # 단계 3. Level-1 메타 러너 학습
    # ------------------------------------------------------------------
    print("\n[Level-1 메타 러너 학습 시작]")
    meta_learner = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    meta_learner.fit(X_train_meta, y_train)
    
    # ------------------------------------------------------------------
    # 단계 4. 기저 모델들 전체 데이터(X_train, y_train) 재학습 및 최신화
    # ------------------------------------------------------------------
    print("\n[기저 모델들 전체 학습 데이터로 재학습 및 최신화 시작...]")
    MODEL_DIR = Path('ml_engine/models')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print("  - [Level-0] XGBoost 전체 학습 중...")
    best_xgb = XGBClassifier(
        n_estimators=541,
        max_depth=4,
        learning_rate=0.03716146208921116,
        subsample=0.6371106085974699,
        colsample_bytree=0.9923202402440235,
        min_child_weight=1,
        gamma=0.40225152117187357,
        reg_alpha=0.85331092958197,
        reg_lambda=1.0860868437421698,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        enable_categorical=False
    )
    best_xgb.fit(X_train, y_train)
    joblib.dump(best_xgb, MODEL_DIR / 'xgboost_pitch_model.pkl', compress=3)
    
    print("  - [Level-0] LightGBM 전체 학습 중...")
    best_lgb = LGBMClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, verbose=-1)
    best_lgb.fit(X_train, y_train)
    joblib.dump(best_lgb, MODEL_DIR / 'lightgbm_pitch_model.pkl', compress=3)
    
    print("  - [Level-0] CatBoost 전체 학습 중...")
    best_cat = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.1, loss_function='MultiClass', random_seed=42, verbose=100)
    best_cat.fit(X_train, y_train)
    joblib.dump(best_cat, MODEL_DIR / 'catboost_pitch_model.pkl', compress=3)
    
    xgb_test_proba = best_xgb.predict_proba(X_test)
    lgb_test_proba = best_lgb.predict_proba(X_test)
    cat_test_proba = best_cat.predict_proba(X_test)
    
    xgb_f1 = f1_score(y_test, best_xgb.predict(X_test), average='weighted', zero_division=0)
    lgb_f1 = f1_score(y_test, best_lgb.predict(X_test), average='weighted', zero_division=0)
    
    cat_pred = best_cat.predict(X_test)
    if len(cat_pred.shape) > 1 and cat_pred.shape[1] == 1:
        cat_pred = cat_pred.ravel()
    cat_f1 = f1_score(y_test, cat_pred, average='weighted', zero_division=0)
    
    print("\n[기저 모델 단독 2025 F1-Score]")
    print(f"  XGBoost F1:    {xgb_f1:.4f}")
    print(f"  LightGBM F1:   {lgb_f1:.4f}")
    print(f"  CatBoost F1:   {cat_f1:.4f}")
    
    X_test_meta = np.hstack([xgb_test_proba, lgb_test_proba, cat_test_proba])
    y_pred_meta = meta_learner.predict(X_test_meta)
    
    stacking_acc  = accuracy_score(y_test, y_pred_meta)
    stacking_prec = precision_score(y_test, y_pred_meta, average='weighted', zero_division=0)
    stacking_rec  = recall_score(y_test, y_pred_meta,  average='weighted', zero_division=0)
    stacking_f1   = f1_score(y_test, y_pred_meta,      average='weighted', zero_division=0)
    
    print("\n[스태킹 앙상블 성능 결과]")
    print(f"  Accuracy:  {stacking_acc:.4f}")
    print(f"  Precision: {stacking_prec:.4f}")
    print(f"  Recall:    {stacking_rec:.4f}")
    print(f"  F1-Score:  {stacking_f1:.4f} (기저 모델 대비 향상 추이 분석)")
    
    # ------------------------------------------------------------------
    # 단계 5. 아티팩트 패킹 및 저장
    # ------------------------------------------------------------------
    meta_path = MODEL_DIR / 'stacking_meta_learner.pkl'
    paths_path = MODEL_DIR / 'stacking_base_model_paths.pkl'
    
    joblib.dump(meta_learner, meta_path, compress=3)
    
    base_model_paths = {
        'xgb': 'ml_engine/models/xgboost_pitch_model.pkl',
        'lgb': 'ml_engine/models/lightgbm_pitch_model.pkl',
        'cat': 'ml_engine/models/catboost_pitch_model.pkl'
    }
    joblib.dump(base_model_paths, paths_path, compress=3)
    
    print(f"\n스태킹 메타 아티팩트 패킹 완료:")
    print(f"  메타러너 저장 경로: {meta_path} ({meta_path.stat().st_size / 1024:.1f} KB)")
    print(f"  경로 딕셔너리 저장: {paths_path} ({paths_path.stat().st_size / 1024:.1f} KB)")
    
    print("\n### 최종 결과 테이블 데이터 ###")
    print(f"XGB:{xgb_f1:.4f}|LGB:{lgb_f1:.4f}|CAT:{cat_f1:.4f}|STACK:{stacking_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling', type=float, default=1.0, help='데이터 샘플링 비율 (기본값: 1.0)')
    args = parser.parse_args()
    
    train_stacking(sampling_rate=args.sampling)
