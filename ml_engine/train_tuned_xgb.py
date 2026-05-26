import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from ml_engine.train import prepare_training_data, pack_model

MODEL_DIR = Path('ml_engine/models')

def train_tuned_xgb():
    print("[*] 100% 데이터로 최종 튜닝된 XGBoost 학습 준비 중...")
    X_train, X_test, y_train, y_test, feat_names, label_encoder = prepare_training_data(sampling_rate=1.0)
    
    best_params = {
        'n_estimators': 541,
        'max_depth': 4,
        'learning_rate': 0.03716146208921116,
        'subsample': 0.6371106085974699,
        'colsample_bytree': 0.9923202402440235,
        'min_child_weight': 1,
        'gamma': 0.40225152117187357,
        'reg_alpha': 0.85331092958197,
        'reg_lambda': 1.0860868437421698,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss',
        'enable_categorical': False
    }

    print("\n[*] 최적 파라미터 적용 XGBoost 학습 시작...")
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"\n[완료] 튜닝된 XGBoost 최종 F1 (100% 데이터): {f1:.4f} ({f1*100:.2f}%)")
    
    pack_model(model, label_encoder)

if __name__ == "__main__":
    train_tuned_xgb()
