# pyrefly: ignore [missing-import]
import optuna
import joblib
import numpy as np
import warnings
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from ml_engine.train import prepare_training_data

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = Path('ml_engine/models')

def objective(trial, X_train, X_test, y_train, y_test):
    """
    [Optuna Objective 함수]
    - 목적: XGBoost 하이퍼파라미터 탐색 → weighted F1-Score 최대화
    - 평가 기준: Chronological Split (2024 train / 2025 val) 고정
    """
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 200, 1500),
        'max_depth':         trial.suggest_int('max_depth', 4, 12),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'gamma':             trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha':         trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda':        trial.suggest_float('reg_lambda', 0.5, 2.0),
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'mlogloss',
        'verbosity': 0,
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return f1_score(y_test, y_pred, average='weighted', zero_division=0)


def run_tuning(n_trials: int = 100, sampling_rate: float = 0.1):
    """
    [Optuna 튜닝 실행 엔트리포인트]
    """
    print(f"[*] 데이터 로드 중 (샘플링 {sampling_rate*100:.0f}%)...")
    X_train, X_test, y_train, y_test, feat_names, label_encoder = prepare_training_data(
        sampling_rate=sampling_rate
    )
    print(f"[*] Optuna 탐색 시작 (n_trials={n_trials})")
    print(f"    탐색 공간: n_estimators, max_depth, learning_rate 등 9개 파라미터")

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )

    import pandas as pd

    def callback(study, trial):
        print(f"  [Trial {trial.number:03d}] F1: {trial.value*100:.2f}% (Best: {study.best_value*100:.2f}%)")

    study.optimize(
        lambda trial: objective(trial, X_train, X_test, y_train, y_test),
        n_trials=n_trials,
        callbacks=[callback]
    )

    print(f"\n[완료] 총 {len(study.trials)}회 탐색")
    print(f"[최고 F1-Score] {study.best_value:.6f} ({study.best_value*100:.2f}%)")
    print(f"[최적 파라미터]")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # - 최적 파라미터로 최종 모델 재학습 및 패킹
    print("\n[*] 최적 파라미터로 최종 모델 재학습 중...")
    best_model = XGBClassifier(
        **study.best_params,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        verbosity=0
    )
    best_model.fit(X_train, y_train)

    # - 피처 중요도 Top 10 출력
    importances = best_model.feature_importances_
    feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
    print("\n[피처 중요도 Top 10]")
    for i, (feat, imp) in enumerate(feat_imp.head(10).items()):
        print(f"  {i+1}. {feat}: {imp:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / 'xgboost_pitch_model.pkl'
    joblib.dump(best_model, model_path, compress=3)
    joblib.dump(label_encoder, MODEL_DIR / 'label_encoder.pkl', compress=3)

    model_kb = model_path.stat().st_size / 1024
    print(f"[패킹 완료] {model_path} ({model_kb:.1f} KB)")
    print(f"[FastAPI 서빙 즉시 가능]")

    return study


if __name__ == "__main__":
    study = run_tuning(n_trials=100, sampling_rate=0.1)
