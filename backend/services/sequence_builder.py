# ==============================================================================
# MLB PitchFlow AI - 실시간 시계열 추론용 3D 텐서 시퀀스 빌더
# 작성일: 2026-05-29
# 설계 원칙:
#   1. bilstm_scaler.pkl & transformer_scaler.pkl 싱글톤 캐싱 (매 요청마다 로드 금지)
#   2. 직전 투구 이력 5개 미만 시 앞쪽(oldest) 제로 패딩
#   3. count_situation, matchup_type 파생 변수 동일 공식 계산 및 100% 매칭
#   4. 이미 스케일링된 1D 피처 벡터를 슬라이딩 윈도우 캐시로 관리하여 연산 극대화
# ==============================================================================

import logging
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

MODEL_DIR = Path("ml_engine/models")

# ------------------------------------------------------------------------------
# 싱글톤 스케일러 및 피처 로더
# ------------------------------------------------------------------------------
_scalers = {}
_scale_features = {}
_nn_features = {}

def _get_scaler_and_features(model_type: str) -> Tuple[joblib.load, List[str], List[str]]:
    global _scalers, _scale_features, _nn_features
    if model_type not in _scalers:
        scaler_path = MODEL_DIR / f"{model_type}_scaler.pkl"
        scale_feat_path = MODEL_DIR / f"{model_type}_scale_features.pkl"
        
        if not scaler_path.exists() or not scale_feat_path.exists():
            raise FileNotFoundError(
                f"시계열 스케일러 파일 누락: {scaler_path} 또는 {scale_feat_path} — 훈련 선행 필요"
            )
            
        _scalers[model_type] = joblib.load(scaler_path)
        _scale_features[model_type] = joblib.load(scale_feat_path)
        
        if model_type == "bilstm":
            nn_feat_path = MODEL_DIR / "transformer_nn_features.pkl"
            if nn_feat_path.exists():
                _nn_features[model_type] = joblib.load(nn_feat_path)
            else:
                from ml_engine.config import ALLOWED_FEATURES
                _nn_features[model_type] = [
                    f for f in ALLOWED_FEATURES
                    if f not in ['pitcher', 'batter', 'fielder_2', 'game_year', 'game_pk', 'at_bat_number', 'pitch_number']
                ]
        elif model_type == "transformer":
            nn_feat_path = MODEL_DIR / "transformer_nn_features.pkl"
            if not nn_feat_path.exists():
                raise FileNotFoundError(f"Transformer NN 피처 파일 누락: {nn_feat_path}")
            _nn_features[model_type] = joblib.load(nn_feat_path)
            
        logger.info(
            f"[{model_type.upper()}] 스케일러 및 {len(_scale_features[model_type])}개 스케일 피처, "
            f"{len(_nn_features[model_type])}개 NN 입력 피처 싱글톤 로드 완료"
        )
        
    return _scalers[model_type], _scale_features[model_type], _nn_features[model_type]


# ------------------------------------------------------------------------------
# 실시간 인게임 투구 히스토리 슬라이딩 윈도우 캐시
# 구조: {(game_pk, pitcher_id, model_type): [scaled_array_1, scaled_array_2, ...]}
# ------------------------------------------------------------------------------
_game_history: Dict[Tuple[int, int, str], List[np.ndarray]] = {}

def _get_seq_len(model_type: str) -> int:
    path = MODEL_DIR / f"best_params_{model_type}.json"
    if path.exists():
        import json
        try:
            with open(path, "r") as f:
                params = json.load(f)
            return params.get("sequence_length", 5)
        except Exception:
            return 5
    return 5  # fallback

def clear_game_history(game_pk: int = None):
    """특정 경기 혹은 전체 인게임 히스토리 캐시 초기화 (메모리 관리 목적)"""
    global _game_history
    if game_pk is not None:
        keys_to_del = [k for k in _game_history.keys() if k[0] == game_pk]
        for k in keys_to_del:
            _game_history.pop(k, None)
        logger.info(f"인게임 히스토리 캐시 초기화 완료: game_pk={game_pk}")
    else:
        _game_history.clear()
        logger.info("전체 인게임 히스토리 캐시 완전 초기화 완료")


# ------------------------------------------------------------------------------
# build_inference_sequence — predict.py 연동 핵심 진입점
# ------------------------------------------------------------------------------
def build_inference_sequence(
    merged_dict: dict,
    game_pk:     int,
    pitcher_id:  int,
    stand_raw:   str,  # 'R' or 'L'
    model_type:  str = "bilstm",
) -> np.ndarray:
    """
    [실시간 추론용 3D 텐서 시퀀스 빌더]
    1. balls, strikes, stand, p_throws를 바탕으로 count_situation, matchup_type을 즉석 계산하여 병합
    2. 스케일링 대상 피처를 스케일러를 통해 정규화
    3. 스케일링된 피처와 스케일 제외 피처(prev_pitch_1,2,3 등)를 모델 고유 nn_features 순서대로 조합
    4. (game_pk, pitcher_id, model_type) 슬라이딩 캐시에서 직전 N개 프레임을 슬라이스하여 3D 텐서 [1, N, D] 생성
    5. 추론이 끝난 후, 다음 투구 시점을 위해 현재 튜플을 캐시에 추가 (최대 N개 유지)

    Args:
        merged_dict: predict.py에서 API 입력과 캐시 피처를 병합 완료한 dict
        game_pk: 경기 식별 고유 ID
        pitcher_id: 투수 MLB player_id
        stand_raw: 타자 타석 방향 ('R' or 'L')
        model_type: "bilstm" 또는 "transformer"

    Returns:
        np.ndarray: Bi-LSTM 및 Transformer 입력용 3D 텐서 (Shape: [1, N, D])
    """
    scaler, scale_features, nn_features = _get_scaler_and_features(model_type)
    
    # --------------------------------------------------------------------------
    # 단계 1. count_situation 및 matchup_type 즉각 복원
    # --------------------------------------------------------------------------
    balls   = merged_dict.get("balls", 0)
    strikes = merged_dict.get("strikes", 0)
    merged_dict["count_situation"] = balls * 3 + strikes
    
    p_throws  = merged_dict.get("p_throws", "R").upper()
    stand_str = stand_raw.upper()
    matchup_key = f"{stand_str}{p_throws}"
    matchup_map = {"LL": 0, "LR": 1, "RL": 2, "RR": 3}
    merged_dict["matchup_type"] = matchup_map.get(matchup_key, 4)

    # --------------------------------------------------------------------------
    # 단계 2. 부분 피처 스케일 정규화 (스케일 제외 피처 우회)
    # --------------------------------------------------------------------------
    # - 스케일링 대상 피처만 수집하여 변환
    scale_vector = []
    for col in scale_features:
        scale_vector.append(float(merged_dict.get(col, 0.0)))
        
    scale_array = np.array([scale_vector], dtype=np.float32)  # Shape: (1, len(scale_features))
    scaled_values = scaler.transform(scale_array)[0]           # Shape: (len(scale_features),)
    
    # - 역매핑 딕셔너리 구성
    scaled_dict = {col: val for col, val in zip(scale_features, scaled_values)}

    # --------------------------------------------------------------------------
    # 단계 3. 최종 NN 피처 순서로 1D 벡터 구성 (bilstm=62, transformer=32)
    # --------------------------------------------------------------------------
    feat_vector = []
    for col in nn_features:
        if col in scaled_dict:
            feat_vector.append(scaled_dict[col])
        else:
            # 스케일링 제외 대상 (prev_pitch_1,2,3 등)은 원래 수치 그대로 대입
            feat_vector.append(float(merged_dict.get(col, -1.0) if "prev_pitch" in col else merged_dict.get(col, 0.0)))
            
    scaled_array = np.array(feat_vector, dtype=np.float32)  # Shape: (D,)

    # --------------------------------------------------------------------------
    # 단계 4. 직전 N구 슬라이딩 윈도우 구성 (현재 투구 N 이전의 직전 N개 프레임 수집, 앞쪽 제로 패딩)
    # --------------------------------------------------------------------------
    cache_key = (game_pk, pitcher_id, model_type)
    if cache_key not in _game_history:
        _game_history[cache_key] = []
        
    history = _game_history[cache_key]
    seq_len = _get_seq_len(model_type)
    
    # 현재 투구 N 이전의 직전 N개 프레임 수집
    window = history[-seq_len:] if len(history) > 0 else []
    
    # 앞쪽 제로 패딩 (Zero Padding)
    if len(window) < seq_len:
        pad_size = seq_len - len(window)
        # 패딩 차원은 NN 피처 개수(D)여야 함
        pad = [np.zeros(len(nn_features), dtype=np.float32) for _ in range(pad_size)]
        sequence = np.vstack(pad + window) if len(window) > 0 else np.vstack(pad)
    else:
        sequence = np.vstack(window)  # Shape: (seq_len, D)
        
    # Batch 차원 추가 -> Shape: (1, seq_len, D)
    inference_tensor = np.expand_dims(sequence, axis=0)

    # --------------------------------------------------------------------------
    # 단계 5. 다음 투구 시점 추론을 위해 현재 프레임을 히스토리에 추가
    # --------------------------------------------------------------------------
    history.append(scaled_array)
    if len(history) > seq_len:
        _game_history[cache_key] = history[-seq_len:]
        
    logger.debug(
        f"3D 시퀀스 구성 완료 ({model_type.upper()}): game_pk={game_pk}, pitcher={pitcher_id}, "
        f"현재 이력 누적 수: {len(history)}구, Tensor Shape: {inference_tensor.shape}"
    )
    
    return inference_tensor
