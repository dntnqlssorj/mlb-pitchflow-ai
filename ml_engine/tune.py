import sys
import gc
import json
import argparse
import warnings
import optuna
import joblib
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from ml_engine.train import prepare_training_data
from ml_engine.sequence_dataset import build_sequence_dataset
from ml_engine.bilstm_model import PitchBiLSTM
from ml_engine.transformer_model import PitchTransformer
from ml_engine.config import ALLOWED_FEATURES

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_DIR = Path('ml_engine/models')
ID_FEATURES = ['pitcher', 'batter', 'fielder_2', 'game_year']

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

# ------------------------------------------------------------------
# [기존] XGBoost Objective
# ------------------------------------------------------------------
from xgboost import XGBClassifier

def objective_xgb(trial, X_train, X_test, y_train, y_test):
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

# ------------------------------------------------------------------
# [신규] Bi-LSTM Objective (과적합 보정 공간 및 sequence_length 반영)
# ------------------------------------------------------------------
def objective_bilstm(trial, df, label_encoder):
    device = get_device()
    n_classes = len(label_encoder.classes_)

    # 1. 과적합 진단에 기초한 최적 HPO 탐색 범위
    hidden_size      = trial.suggest_categorical('hidden_size', [32, 64, 128])
    num_layers       = trial.suggest_int('num_layers', 1, 3)
    dropout          = trial.suggest_float('dropout', 0.2, 0.5)
    learning_rate    = trial.suggest_float('learning_rate', 5e-5, 1e-3, log=True)
    batch_size       = trial.suggest_categorical('batch_size', [128, 256, 512])
    sequence_length  = trial.suggest_categorical('sequence_length', [5, 7, 10])

    scale_features = [f for f in ALLOWED_FEATURES if f in df.columns and f not in ID_FEATURES]
    scaler = StandardScaler()
    train_mask = df['game_year'] == 2024
    
    df_copy = df.copy()
    scaler.fit(df_copy.loc[train_mask, scale_features])
    df_copy[scale_features] = scaler.transform(df_copy[scale_features])

    nn_features = [f for f in ALLOWED_FEATURES if f not in ID_FEATURES]
    feature_dim = len(nn_features)

    train_dataset = build_sequence_dataset(df_copy, label_encoder, split='train', features=nn_features, seq_len=sequence_length)
    val_dataset   = build_sequence_dataset(df_copy, label_encoder, split='test', features=nn_features, seq_len=sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = PitchBiLSTM(
        feature_dim=feature_dim, n_classes=n_classes,
        hidden_size=hidden_size, num_layers=num_layers, dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 3 Epochs 제한 훈련 및 Pruning 연동
    for epoch in range(1, 4):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds.append(model(xb).argmax(1).cpu().numpy())
                all_labels.append(yb.numpy())
        
        preds = np.concatenate(preds)
        all_labels = np.concatenate(all_labels)
        val_f1 = f1_score(all_labels, preds, average='weighted', zero_division=0)

        # Optuna Pruning 보고
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_f1

# ------------------------------------------------------------------
# [신규] Transformer Objective (sequence_length 반영)
# ------------------------------------------------------------------
def objective_transformer(trial, df, label_encoder):
    device = get_device()
    n_classes = len(label_encoder.classes_)

    # 1. 튜닝 파라미터 샘플링 공간 정의
    d_model         = trial.suggest_categorical('d_model', [32, 64, 128])
    nhead           = trial.suggest_categorical('nhead', [2, 4, 8])
    
    # [조건 엄수 6] d_model % nhead != 0 이면 강제 TrialPruned
    if d_model % nhead != 0:
        raise optuna.exceptions.TrialPruned()

    num_layers       = trial.suggest_int('num_layers', 1, 3)
    dim_feedforward  = trial.suggest_categorical('dim_feedforward', [64, 128, 256])
    dropout          = trial.suggest_float('dropout', 0.05, 0.3)
    learning_rate    = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    batch_size       = trial.suggest_categorical('batch_size', [256, 512])
    sequence_length  = trial.suggest_categorical('sequence_length', [5, 7, 10])

    scale_features = [f for f in ALLOWED_FEATURES if f in df.columns and f not in ID_FEATURES]
    scaler = StandardScaler()
    train_mask = df['game_year'] == 2024
    
    df_copy = df.copy()
    scaler.fit(df_copy.loc[train_mask, scale_features])
    df_copy[scale_features] = scaler.transform(df_copy[scale_features])

    nn_features = [f for f in ALLOWED_FEATURES if f not in ID_FEATURES]
    feature_dim = len(nn_features)

    train_dataset = build_sequence_dataset(df_copy, label_encoder, split='train', features=nn_features, seq_len=sequence_length)
    val_dataset   = build_sequence_dataset(df_copy, label_encoder, split='test', features=nn_features, seq_len=sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = PitchTransformer(
        feature_dim=feature_dim, n_classes=n_classes,
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        dim_feedforward=dim_feedforward, dropout=dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # 3 Epochs 제한 훈련 및 Pruning 연동
    for epoch in range(1, 4):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds.append(model(xb).argmax(1).cpu().numpy())
                all_labels.append(yb.numpy())
        
        preds = np.concatenate(preds)
        all_labels = np.concatenate(all_labels)
        val_f1 = f1_score(all_labels, preds, average='weighted', zero_division=0)

        # Optuna Pruning 보고
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_f1

# ------------------------------------------------------------------
# 최종 전체 100% 데이터 학습용 러너 (가변 sequence_length 수렴 지원)
# ------------------------------------------------------------------
def run_final_full_train(model_type, best_params):
    device = get_device()
    print(f"\n[*] [전체 재학습 시작] 모델 타입: {model_type.upper()}")
    sys.stdout.flush()

    # 1. 100% 전체 데이터 로드
    _, _, _, _, feat_names, label_encoder, df = prepare_training_data(sampling_rate=1.0, return_df=True)
    n_classes = len(label_encoder.classes_)

    # 2. 스케일링 적용
    scale_features = [f for f in ALLOWED_FEATURES if f in df.columns and f not in ID_FEATURES]
    scaler = StandardScaler()
    train_mask = df['game_year'] == 2024
    scaler.fit(df.loc[train_mask, scale_features])
    df[scale_features] = scaler.transform(df[scale_features])

    nn_features = [f for f in ALLOWED_FEATURES if f not in ID_FEATURES]
    feature_dim = len(nn_features)

    sequence_length = best_params.get('sequence_length', 5)
    train_dataset = build_sequence_dataset(df, label_encoder, split='train', features=nn_features, seq_len=sequence_length)
    val_dataset   = build_sequence_dataset(df, label_encoder, split='test', features=nn_features, seq_len=sequence_length)

    batch_size = best_params.get('batch_size', 256)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if model_type == 'bilstm':
        model = PitchBiLSTM(
            feature_dim=feature_dim, n_classes=n_classes,
            hidden_size=best_params['hidden_size'],
            num_layers=best_params['num_layers'],
            dropout=best_params['dropout']
        ).to(device)
        learning_rate = best_params.get('learning_rate', best_params.get('lr', 3e-4))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        save_path = MODEL_DIR / 'bilstm_pitch_model.pt'
        scaler_path = MODEL_DIR / 'bilstm_scaler.pkl'
        sfeats_path = MODEL_DIR / 'bilstm_scale_features.pkl'
        
    elif model_type == 'transformer':
        model = PitchTransformer(
            feature_dim=feature_dim, n_classes=n_classes,
            d_model=best_params['d_model'],
            nhead=best_params['nhead'],
            num_layers=best_params['num_layers'],
            dim_feedforward=best_params['dim_feedforward'],
            dropout=best_params['dropout']
        ).to(device)
        learning_rate = best_params.get('learning_rate', best_params.get('lr', 1e-3))
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        save_path = MODEL_DIR / 'transformer_pitch_model.pt'
        scaler_path = MODEL_DIR / 'transformer_scaler.pkl'
        sfeats_path = MODEL_DIR / 'transformer_scale_features.pkl'
        nnfeats_path = MODEL_DIR / 'transformer_nn_features.pkl'
        joblib.dump(nn_features, nnfeats_path, compress=3)

    criterion = nn.CrossEntropyLoss()
    EPOCHS = 15 # 최적화 완료 후 안정적인 전체 데이터 수렴을 위한 15 Epochs 고정
    best_f1 = 0.0

    print(f"[*] {EPOCHS} Epochs 최종 전체 학습 시작...")
    sys.stdout.flush()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds.append(model(xb).argmax(1).cpu().numpy())
                all_labels.append(yb.numpy())
        
        preds = np.concatenate(preds)
        all_labels = np.concatenate(all_labels)
        val_f1 = f1_score(all_labels, preds, average='weighted', zero_division=0)
        print(f"  [Epoch {epoch:02d}/{EPOCHS}] 검증 F1: {val_f1*100:.2f}%")
        sys.stdout.flush()

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            joblib.dump(scaler, scaler_path, compress=3)
            joblib.dump(scale_features, sfeats_path, compress=3)
            # sequence_length 메타 저장 자동화 (추론 연계 대응)
            if model_type == 'bilstm':
                joblib.dump({'sequence_length': sequence_length}, MODEL_DIR / 'bilstm_nn_idx.pkl', compress=3)
            elif model_type == 'transformer':
                joblib.dump({'sequence_length': sequence_length}, MODEL_DIR / 'transformer_nn_idx.pkl', compress=3)
            print(f"  → Best 모델 저장 완료 (F1: {best_f1*100:.2f}%)")
            sys.stdout.flush()

    print(f"\n[전체 재학습 완료] 최적 F1: {best_f1*100:.2f}%")
    sys.stdout.flush()

# ------------------------------------------------------------------
# 메인 제어 루프
# ------------------------------------------------------------------
def run_tuning_main(model_type: str, n_trials: int = 50):
    print(f"\n==================================================")
    print(f"[*] Optuna HPO 기동: 모델={model_type.upper()}, Trials={n_trials}")
    print(f"==================================================")
    sys.stdout.flush()

    # 데이터셋 준비 (10% 샘플링 고정)
    print("[*] HPO 데이터 가공 로딩 시작 (10% 샘플링)...")
    sys.stdout.flush()
    _, _, _, _, feat_names, label_encoder, df = prepare_training_data(sampling_rate=0.1, return_df=True)

    # Pruner 및 Sampler 정의
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

    def callback(study, trial):
        print(f"  [Trial {trial.number:03d}] F1: {trial.value*100:.2f}% (Best: {study.best_value*100:.2f}%)")
        sys.stdout.flush()

    if model_type == 'xgb':
        # 기존 XGBoost 흐름
        X_train, X_test, y_train, y_test, _, _ = prepare_training_data(sampling_rate=0.1)
        study.optimize(lambda t: objective_xgb(t, X_train, X_test, y_train, y_test), n_trials=n_trials, callbacks=[callback])
        
        # 저장
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with open(MODEL_DIR / 'best_params_xgb.json', 'w') as f:
            json.dump(study.best_params, f, indent=4)
            
    elif model_type == 'bilstm':
        study.optimize(lambda t: objective_bilstm(t, df, label_encoder), n_trials=n_trials, callbacks=[callback])
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        param_path = MODEL_DIR / 'best_params_bilstm.json'
        with open(param_path, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        print(f"[최적화 파라미터 JSON 저장] {param_path}")
        
        # 전체 데이터 최종 재학습 기동
        run_final_full_train('bilstm', study.best_params)

    elif model_type == 'transformer':
        study.optimize(lambda t: objective_transformer(t, df, label_encoder), n_trials=n_trials, callbacks=[callback])
        
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        param_path = MODEL_DIR / 'best_params_transformer.json'
        with open(param_path, 'w') as f:
            json.dump(study.best_params, f, indent=4)
        print(f"[최적화 파라미터 JSON 저장] {param_path}")
        
        # 전체 데이터 최종 재학습 기동
        run_final_full_train('transformer', study.best_params)

    print(f"\n🏆 HPO 탐색 최종 결과: Best Val F1 = {study.best_value*100:.2f}%")
    sys.stdout.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='xgb', choices=['xgb', 'bilstm', 'transformer'], help='HPO 대상 모델')
    parser.add_argument('--n_trials', type=int, default=50, help='Optuna 탐색 횟수')
    args = parser.parse_args()

    run_tuning_main(model_type=args.model, n_trials=args.n_trials)
