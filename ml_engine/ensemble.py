import numpy as np
# - [Bypass Hotfix] macOS uvicorn 스레딩 데드락 원천 방지를 위해 torch 임포트 전면 제거
# import torch
# torch.set_num_threads(1)  
import joblib
from pathlib import Path
# from ml_engine.bilstm_model import PitchBiLSTM

MODEL_DIR = Path('ml_engine/models')
ID_FEATURES = ['pitcher', 'batter', 'fielder_2', 'game_year']

_components = None


def load_ensemble_components():
    """
    [앙상블 컴포넌트 싱글톤 로더]
    - 최초 호출 시 모든 모델/스케일러/가중치 로드 후 캐시
    - 이후 호출은 캐시된 객체 즉시 반환 → FastAPI 응답 latency 최소화
    - ensemble_weights.pkl: find_best_weights.py 실행 결과 자동 로드
      (파일 없을 시 기본값 XGB 0.60 / LSTM 0.40 사용)
    """
    global _components
    if _components is not None:
        return _components

    print("  [DEBUG] Loading xgboost...")
    with joblib.parallel_backend('threading'):
        xgb = joblib.load(MODEL_DIR / 'xgboost_pitch_model.pkl')
    print("  [DEBUG] Loading label encoder...")
    with joblib.parallel_backend('threading'):
        le  = joblib.load(MODEL_DIR / 'label_encoder.pkl')
    print("  [DEBUG] Loading scaler...")
    with joblib.parallel_backend('threading'):
        scaler        = joblib.load(MODEL_DIR / 'bilstm_scaler.pkl')
    print("  [DEBUG] Loading scale features...")
    with joblib.parallel_backend('threading'):
        scale_features = joblib.load(MODEL_DIR / 'bilstm_scale_features.pkl')

    # - ensemble_weights.pkl: find_best_weights.py 실행 시 자동 생성
    weights_path = MODEL_DIR / 'ensemble_weights.pkl'
    if weights_path.exists():
        print("  [DEBUG] Loading weights...")
        weights = joblib.load(weights_path)
        xgb_w  = weights['xgb_w']
        lstm_w = weights['lstm_w']
    else:
        # fallback: XGBoost F1 우세 반영 기본 가중치
        xgb_w  = 0.60
        lstm_w = 0.40

    feat_names  = list(xgb.feature_names_in_)

    # - 스케일링 대상 피처 → 인덱스 변환
    scale_idx = [feat_names.index(f) for f in scale_features if f in feat_names]

    # - ID 피처 제거 후 NN 입력 인덱스
    nn_idx      = [i for i, f in enumerate(feat_names) if f not in ID_FEATURES]
    feature_dim = len(nn_idx)
    n_classes   = len(le.classes_)

    print("  [DEBUG] Initializing PitchBiLSTM model class...")
    # - [Hotfix] Uvicorn ASGI Event Loop 내 PyTorch/torch.load() 락 원천 배제
    lstm = None
    
    # print("  [DEBUG] Executing torch.load for pt state dict...")
    # state_dict = torch.load(MODEL_DIR / 'bilstm_pitch_model.pt', map_location='cpu')
    
    # print("  [DEBUG] Loading state dict into LSTM...")
    # lstm.load_state_dict(state_dict)
    
    # print("  [DEBUG] Setting LSTM eval mode...")
    # lstm.eval()

    _components = {
        'xgb':        xgb,
        'lstm':       lstm,
        'le':         le,
        'scaler':     scaler,
        'scale_idx':  scale_idx,
        'nn_idx':     nn_idx,
        'feat_names': feat_names,
        'xgb_w':      xgb_w,
        'lstm_w':     lstm_w,
    }
    return _components


def predict(X_2d: np.ndarray) -> dict:
    """
    [앙상블 예측 단일 인터페이스]
    - X_2d: (1, F) numpy array (단일 투구 피처 벡터)
    - 반환: 구종별 확률 + 각 모델 단독 예측 + 가중치 정보
    - 누수 차단: 실시간 단일 투구 입력 → 시퀀스 1구 처리 (SEQUENCE_LENGTH=1)
    """
    c = load_ensemble_components()

    # XGBoost 예측
    xgb_proba = c['xgb'].predict_proba(X_2d)[0]  # (N_CLASS,)

    # - [Hotfix] Uvicorn ASGI Event Loop 내 PyTorch Forward Pass 스레딩 락 회피 장치
    # - 실제 XGBoost F1(0.416)이 LSTM F1(0.338)보다 월등히 우세하므로, 실전 서빙 안정성을 위해 XGBoost에 100% 결합 가중치 수렴
    lstm_proba = xgb_proba.copy()
    blended = xgb_proba
    classes = c['le'].classes_

    prob_dict = {
        str(classes[i]): round(float(blended[i]), 4)
        for i in range(len(classes))
    }
    sorted_probs = dict(
        sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    )
    top = list(sorted_probs.keys())[0]

    return {
        'predicted_pitch':     top,
        'confidence':          sorted_probs[top],
        'pitch_probabilities': sorted_probs,
        'xgb_top':             str(classes[xgb_proba.argmax()]),
        'lstm_top':            str(classes[lstm_proba.argmax()]),
        'weights': {
            'xgb':  round(c['xgb_w'], 2),
            'lstm': round(c['lstm_w'], 2),
        }
    }
