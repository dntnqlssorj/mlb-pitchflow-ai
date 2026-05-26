# ==============================================================================
# MLB PitchFlow AI - 구종 예측 API 라우터
# 변경 이력: 2026-05-21 API 입력 스키마 축소 및 enrichment 서비스 연동
#   - PitchInferenceInput: 사후 물리 지표 전면 제거, 9-식별자 + 경기 상황으로 축소
#   - predict_pitch: enrich_pitch_context() 호출 후 모델 입력 딕셔너리 자동 조립
#   - _build_feature_vector: dict 입력으로 시그니처 변경 (Pydantic 모델 의존 제거)
#   - 응답에 enrichment_latency_ms, enrichment_sources 추가
# ==============================================================================

import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional

from backend.services.enrichment import enrich_pitch_context
from ml_engine.ensemble import predict as ensemble_predict

router = APIRouter()

MODEL_DIR = Path("ml_engine/models")

MODEL_MAP = {
    "xgboost":       "xgboost_pitch_model.pkl",
    "random_forest": "random_forest_pitch_model.pkl",
    "lightgbm":      "lightgbm_pitch_model.pkl",
    # ensemble: ml_engine/ensemble.py 경유 (별도 pkl 없음)
}


# ==============================================================================
# 입력 스키마 — 9-식별자 + 경기 상황 (사후 물리 지표 전면 제거)
# ==============================================================================
class PitchInferenceInput(BaseModel):
    """
    [추론 요청 데이터 스키마]
    - 설계 원칙 (plan.md §5.1):
        사후 물리 지표 전면 제거. API 호출자는 투구 이전 시점에 알 수 있는
        식별자와 경기 상황만 전송. enrichment.py가 Supabase 조회 후 나머지 자동 조립.
    - 제거된 필드 (완전 삭제):
        release_speed, release_spin_rate, spin_axis, release_pos_x/y/z,
        release_extension, arm_angle, effective_speed,
        api_break_z_with_gravity, api_break_x_arm, api_break_x_batter_in,
        pfx_x, pfx_z, plate_x, plate_z,
        stamina_index, velocity_decay_ratio, spin_decay_ratio,
        base_speed, base_spin, blocking_leverage_factor,
        catcher_blocking_runs, team_oaa_total, fielding_risk_index, is_risp
    """

    # ------------------------------------------------------------------
    # 그룹 A: 선수 식별자 (5개 필수 + 야수 7개 선택)
    # ------------------------------------------------------------------
    pitcher:   int = Field(..., description="투수 MLB player_id")
    batter:    int = Field(..., description="타자 MLB player_id")
    fielder_2: int = Field(..., description="포수 MLB player_id")

    fielder_3: int = Field(0, description="1루수 MLB player_id (없으면 0)")
    fielder_4: int = Field(0, description="2루수 MLB player_id (없으면 0)")
    fielder_5: int = Field(0, description="3루수 MLB player_id (없으면 0)")
    fielder_6: int = Field(0, description="유격수 MLB player_id (없으면 0)")
    fielder_7: int = Field(0, description="좌익수 MLB player_id (없으면 0)")
    fielder_8: int = Field(0, description="중견수 MLB player_id (없으면 0)")
    fielder_9: int = Field(0, description="우익수 MLB player_id (없으면 0)")

    # ------------------------------------------------------------------
    # 그룹 B: 경기 식별자
    # ------------------------------------------------------------------
    game_pk:   int = Field(..., description="경기 고유 ID (Supabase enrichment 조회 키)")
    game_year: int = Field(2025, description="시즌 연도 (기본값: 2025)")

    # ------------------------------------------------------------------
    # 그룹 C: 경기 상황 — 투구 이전 시점 데이터 (필수)
    # ------------------------------------------------------------------
    balls:        int = Field(..., ge=0, le=3,  description="볼 카운트 (0~3)")
    strikes:      int = Field(..., ge=0, le=2,  description="스트라이크 카운트 (0~2)")
    outs_when_up: int = Field(..., ge=0, le=2,  description="현재 아웃 수 (0~2)")
    inning:       int = Field(..., ge=1,        description="이닝 (1~)")

    on_1b: int = Field(0, description="1루 주자 player_id (없으면 0)")
    on_2b: int = Field(0, description="2루 주자 player_id (없으면 0)")
    on_3b: int = Field(0, description="3루 주자 player_id (없으면 0)")

    stand: str = Field("R", description="타자 타석 방향 (R 또는 L)")

    # ------------------------------------------------------------------
    # 그룹 D: enrichment 보조 입력 (선택 — 없으면 DB에서 자동 산출)
    # ------------------------------------------------------------------
    pitch_count_override: Optional[int] = Field(
        None, description="직접 입력 누적 투구 수 (n8n 적재 딜레이 회피용)"
    )
    home_score_diff: int   = Field(0,   description="현재 점수차 (home - away)")
    bat_score_diff:  int   = Field(0,   description="타자 팀 점수차")
    n_thruorder_pitcher:             Optional[int]   = Field(None, description="타순 순환 횟수")
    pitcher_days_since_prev_game:    Optional[int]   = Field(None, description="직전 등판 후 경과일")
    batter_days_since_prev_game:     Optional[int]   = Field(None, description="타자 직전 경기 후 경과일")
    n_priorpa_thisgame_player_at_bat: Optional[int]  = Field(None, description="해당 경기 타자 누적 타석 수")
    age_pit: Optional[int] = Field(None, description="투수 나이")
    age_bat: Optional[int] = Field(None, description="타자 나이")


# ==============================================================================
# 내부 헬퍼 함수
# ==============================================================================
def _load_model(model_type: str):
    """model_type 기반 pkl 동적 로드. 변경 없음."""
    if model_type not in MODEL_MAP:
        raise HTTPException(
            status_code=404,
            detail=f"모델 '{model_type}'을 찾을 수 없습니다. 사용 가능: {list(MODEL_MAP.keys())}"
        )
    model_path = MODEL_DIR / MODEL_MAP[model_type]
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"모델 파일이 아직 학습되지 않았습니다: {model_path}"
        )
    return joblib.load(model_path)


def _load_label_encoder():
    """라벨 인코더 역직렬화. 변경 없음."""
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    if not encoder_path.exists():
        raise HTTPException(status_code=503, detail="라벨 인코더 파일이 존재하지 않습니다.")
    return joblib.load(encoder_path)


def _build_feature_vector(merged_dict: dict, feature_names: list) -> pd.DataFrame:
    """
    [피처 벡터 구성 — 시그니처 변경]
    - 변경 전: input_data: PitchInferenceInput (Pydantic 모델)
    - 변경 후: merged_dict: dict (API 입력 + enriched 딕셔너리 병합 결과)
    - 이유: enrichment 결과가 Pydantic 모델에 없는 키를 포함하므로 dict로 통합 처리

    학습 피처 목록(feature_names) 기준으로 자동 정렬 및 결측 컬럼 0 패딩.
    """
    df = pd.DataFrame([merged_dict])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names].fillna(0)


# ==============================================================================
# 예측 엔드포인트
# ==============================================================================
@router.post(
    "/pitch",
    summary="다음 투구 구종 예측",
    description=(
        "투구 이전 경기 상황 식별자만 입력하면 Supabase 마스터 테이블 조회 후 "
        "자동 피처 조립을 거쳐 다음 투구 구종 확률 분포를 반환합니다."
    ),
    tags=["Prediction"],
)
def predict_pitch(
    input_data: PitchInferenceInput,
    model_type: str = Query(
        default="xgboost",
        description="사용할 모델 (xgboost / random_forest / lightgbm / ensemble)",
    ),
):
    """
    [구종 예측 엔드포인트 — enrichment 연동 버전]
    흐름:
        입력 수신 (9-식별자 + 경기 상황)
        → 모델 로드
        → enrich_pitch_context() [Supabase 조회 + 도메인 피처 조립]
        → 입력 딕셔너리 병합
        → _build_feature_vector() [학습 피처 정렬 + 0 패딩]
        → predict_proba
        → 라벨 디코딩
        → JSON 반환
    """
    # --- Enrichment 공통 처리 (ensemble 포함 모든 모델) ---
    stand_encoded = 0 if input_data.stand.upper() != "L" else 1
    fielder_ids = [
        input_data.fielder_3, input_data.fielder_4, input_data.fielder_5,
        input_data.fielder_6, input_data.fielder_7, input_data.fielder_8,
        input_data.fielder_9,
    ]
    enriched = enrich_pitch_context(
        pitcher_id=input_data.pitcher,
        batter_id=input_data.batter,
        catcher_id=input_data.fielder_2,
        fielder_ids=fielder_ids,
        game_pk=input_data.game_pk,
        game_year=input_data.game_year,
        on_2b=input_data.on_2b,
        on_3b=input_data.on_3b,
        pitch_count_override=input_data.pitch_count_override,
    )
    enrichment_latency_ms = enriched.pop("enrichment_latency_ms")
    enrichment_sources    = enriched.pop("enrichment_sources")

    api_dict = {
        "pitcher":    input_data.pitcher,
        "batter":     input_data.batter,
        "fielder_2":  input_data.fielder_2,
        "fielder_3":  input_data.fielder_3,
        "fielder_4":  input_data.fielder_4,
        "fielder_5":  input_data.fielder_5,
        "fielder_6":  input_data.fielder_6,
        "fielder_7":  input_data.fielder_7,
        "fielder_8":  input_data.fielder_8,
        "fielder_9":  input_data.fielder_9,
        "game_year":  input_data.game_year,
        "balls":      input_data.balls,
        "strikes":    input_data.strikes,
        "outs_when_up": input_data.outs_when_up,
        "inning":     input_data.inning,
        "on_1b":      input_data.on_1b,
        "on_2b":      input_data.on_2b,
        "on_3b":      input_data.on_3b,
        "stand":      stand_encoded,
        "home_score_diff": input_data.home_score_diff,
        "bat_score_diff":  input_data.bat_score_diff,
        "n_thruorder_pitcher":              input_data.n_thruorder_pitcher or 0,
        "pitcher_days_since_prev_game":     input_data.pitcher_days_since_prev_game or 0,
        "batter_days_since_prev_game":      input_data.batter_days_since_prev_game or 0,
        "n_priorpa_thisgame_player_at_bat": input_data.n_priorpa_thisgame_player_at_bat or 0,
        "age_pit": input_data.age_pit or 0,
        "age_bat": input_data.age_bat or 0,
    }
    merged = {**api_dict, **enriched}

    # ------------------------------------------------------------------
    # 앙상블 분기 — XGBoost + Bi-LSTM Soft Blending
    # ------------------------------------------------------------------
    if model_type == "ensemble":
        # XGBoost feature_names 기준으로 피처 벡터 구성
        from ml_engine.ensemble import load_ensemble_components
        c = load_ensemble_components()
        X_df = _build_feature_vector(merged, c['feat_names'])
        X_2d = X_df.values.astype(np.float32)

        result = ensemble_predict(X_2d)
        return {
            "model_used":           "ensemble",
            "predicted_pitch":      result['predicted_pitch'],
            "confidence":           result['confidence'],
            "pitch_probabilities":  result['pitch_probabilities'],
            "xgb_top":             result['xgb_top'],
            "lstm_top":            result['lstm_top'],
            "ensemble_weights":    result['weights'],
            "enrichment_latency_ms": enrichment_latency_ms,
            "enrichment_sources":  enrichment_sources,
        }

    # --- 단일 모델 분기 (xgboost / random_forest / lightgbm) ---
    model         = _load_model(model_type)
    label_encoder = _load_label_encoder()
    feature_names = list(model.feature_names_in_)

    # --- 피처 벡터 구성 및 예측 (단일 모델) ---
    X = _build_feature_vector(merged, feature_names)

    probabilities = model.predict_proba(X)[0]
    pitch_classes = label_encoder.classes_
    prob_dict     = {
        str(pitch_classes[i]): round(float(p), 4)
        for i, p in enumerate(probabilities)
    }

    sorted_probs    = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    predicted_pitch = max(prob_dict, key=prob_dict.get)

    return {
        "model_used":           model_type,
        "predicted_pitch":      predicted_pitch,
        "confidence":           sorted_probs[predicted_pitch],
        "pitch_probabilities":  sorted_probs,
        "enrichment_latency_ms": enrichment_latency_ms,
        "enrichment_sources":   enrichment_sources,
    }