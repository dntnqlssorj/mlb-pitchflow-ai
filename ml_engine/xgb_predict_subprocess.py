"""
[XGBoost 격리 예측 서브프로세스]
- 목적: prepare_training_data() 메모리 선점 후 XGBoost C++ 역직렬화 시
  Python 3.14에서 발생하는 SIGSEGV (KERN_INVALID_ADDRESS 0x580) 회피
- 실행: find_best_weights.py가 subprocess.run()으로 호출 (fresh 메모리에서 시작)
- 입력: /tmp/X_test_raw_ensemble.npy + /tmp/X_test_cols_ensemble.json
- 출력: /tmp/xgb_probas_ensemble.npy
"""
import sys
import json
import numpy as np
import joblib
import xgboost
from pathlib import Path

MODEL_DIR   = Path('ml_engine/models')
INPUT_NPY   = '/tmp/X_test_raw_ensemble.npy'
INPUT_COLS  = '/tmp/X_test_cols_ensemble.json'
OUTPUT_PATH = '/tmp/xgb_probas_ensemble.npy'

print('[XGB-SUB] 모델 로드 중...', flush=True)
xgb        = joblib.load(MODEL_DIR / 'xgboost_pitch_model.pkl')
feat_names = list(xgb.feature_names_in_)
booster    = xgb.get_booster()
print(f'[XGB-SUB] 로드 완료 ({len(feat_names)}개 피처)', flush=True)

# 입력 데이터 로드 (npy + 컬럼명 JSON)
print('[XGB-SUB] 입력 데이터 로드 중...', flush=True)
X_all  = np.load(INPUT_NPY)                          # (N, all_cols)
all_cols = json.load(open(INPUT_COLS))               # 컬럼명 리스트

# XGB feat_names 순서로 컬럼 추출
col_map     = {c: i for i, c in enumerate(all_cols)}
feat_indices = [col_map[f] for f in feat_names]
X_test      = X_all[:, feat_indices]
print(f'[XGB-SUB] 데이터 준비 완료: shape={X_test.shape}', flush=True)

# DMatrix 생성 및 예측
dm     = xgboost.DMatrix(X_test, feature_names=feat_names)
probas = booster.predict(dm)
print(f'[XGB-SUB] 예측 완료: shape={probas.shape}', flush=True)

np.save(OUTPUT_PATH, probas)
print(f'[XGB-SUB] {OUTPUT_PATH} 저장 완료', flush=True)
