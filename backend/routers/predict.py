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
from backend.services.scouting_predictor import predict_with_scouting_llm

router = APIRouter()

MODEL_DIR = Path("ml_engine/models")

MODEL_MAP = {
    "xgboost":       "xgboost_pitch_model.pkl",
    "random_forest": "random_forest_pitch_model.pkl",
    "lightgbm":      "lightgbm_pitch_model.pkl",
    "catboost":      "catboost_pitch_model.pkl",
    # ensemble: ml_engine/ensemble.py 경유 (별도 pkl 없음)
}

# --- PyTorch 시계열 모델 싱글톤 캐시 로더 ---
_pytorch_models = {}

# --- Stacking 모델 싱글톤 캐시 ---
_stacking_cache: dict = {}

# --- Per-Pitcher 로컬 모델 싱글톤 캐시 ---
_per_pitcher_cache: dict = {}
LOCAL_DIR = MODEL_DIR / "local"

def _get_pytorch_model(model_type: str):
    global _pytorch_models
    if model_type in _pytorch_models:
        return _pytorch_models[model_type]

    import json
    import torch
    from pathlib import Path

    MODEL_DIR = Path("ml_engine/models")
    params_path = MODEL_DIR / f"best_params_{model_type}.json"
    params = {}
    if params_path.exists():
        with open(params_path, "r") as f:
            params = json.load(f)

    # feature_dim 확인 (bilstm과 transformer는 동일한 71차원 NN 피처를 공유하므로, bilstm_nn_features.pkl가 없으면 transformer_nn_features.pkl를 공용 로드합니다)
    nn_feat_path = MODEL_DIR / f"{model_type}_nn_features.pkl"
    if not nn_feat_path.exists():
        nn_feat_path = MODEL_DIR / "transformer_nn_features.pkl"

    if not nn_feat_path.exists():
        from ml_engine.config import ALLOWED_FEATURES
        nn_features = [f for f in ALLOWED_FEATURES
                       if f not in ['pitcher','batter','fielder_2','game_year',
                                    'game_pk','at_bat_number','pitch_number']]
    else:
        import joblib
        nn_features = joblib.load(nn_feat_path)
    feature_dim = len(nn_features)

    model_path = MODEL_DIR / f"{model_type}_pitch_model.pt"

    # label encoder로 n_classes 확인하되, 실제 저장된 모델 가중치 체크포인트의 최종 레이어 bias 크기를 확인하여 정합성을 일치시킵니다.
    import joblib
    le = joblib.load(MODEL_DIR / "label_encoder.pkl")
    n_classes = len(le.classes_)
    
    if model_path.exists():
        try:
            checkpoint_sd = torch.load(model_path, map_location="cpu")
            if "classifier.3.bias" in checkpoint_sd:
                n_classes = checkpoint_sd["classifier.3.bias"].shape[0]
        except Exception:
            pass

    if model_type == "bilstm":
        from ml_engine.bilstm_model import PitchBiLSTM
        model = PitchBiLSTM(
            feature_dim=feature_dim,
            n_classes=n_classes,
            hidden_size=params.get("hidden_size", 128),
            num_layers=params.get("num_layers", 2),
            dropout=params.get("dropout", 0.3),
        )
    elif model_type == "transformer":
        from ml_engine.transformer_model import PitchTransformer
        model = PitchTransformer(
            feature_dim=feature_dim,
            n_classes=n_classes,
            d_model=params.get("d_model", 64),
            nhead=params.get("nhead", 4),
            num_layers=params.get("num_layers", 2),
            dim_feedforward=params.get("dim_feedforward", 128),
            dropout=params.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"지원하지 않는 PyTorch 모델 타입: {model_type}")

    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"PyTorch 모델 가중치 파일 없음: {model_path} — 훈련 먼저 수행 필요"
        )

    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    _pytorch_models[model_type] = model
    return model


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
        description="사용할 모델 (xgboost / random_forest / lightgbm / catboost / stacking / ensemble / bilstm / transformer)",
    ),
):
    """
    [구종 예측 엔드포인트 — enrichment 연동 버전]
    """
    global _per_pitcher_cache, _stacking_cache
    stand_encoded = 0 if input_data.stand.upper() != "L" else 1
    fielder_ids = [
        input_data.fielder_3, input_data.fielder_4, input_data.fielder_5,
        input_data.fielder_6, input_data.fielder_7, input_data.fielder_8,
        input_data.fielder_9,
    ]
    print(f"[PREDICT] enrichment start")
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
        inning=input_data.inning,
        balls=input_data.balls,        
        strikes=input_data.strikes,    
        stand=input_data.stand,
    )
    print(f"[PREDICT] enrichment done: {enriched}")
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
    # 시계열 딥러닝 분기 — Bi-LSTM / Transformer 실시간 추론 [신규 추가]
    # ------------------------------------------------------------------
    if model_type in ["bilstm", "transformer"]:
        import torch
        from backend.services.sequence_builder import build_inference_sequence
        
        # 1. 3D 시퀀스 텐서 조립 ([1, 5, D] numpy array)
        X_seq_np = build_inference_sequence(
            merged_dict=merged,
            game_pk=input_data.game_pk,
            pitcher_id=input_data.pitcher,
            stand_raw=input_data.stand,
            model_type=model_type
        )
        
        X_seq_tensor = torch.FloatTensor(X_seq_np)
        
        # 2. 모델 로드 및 모델별 클래스 매핑 설정
        nn_model = _get_pytorch_model(model_type)
        
        label_encoder = _load_label_encoder()
        pitch_classes = label_encoder.classes_
        
        # 3. 추론 실행
        with torch.no_grad():
            logits = nn_model(X_seq_tensor)
            probabilities = torch.softmax(logits, dim=1).numpy()[0]
            
        prob_dict = {
            str(pitch_classes[i]): round(float(p), 4)
            for i, p in enumerate(probabilities)
        }
        
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        predicted_pitch = max(prob_dict, key=prob_dict.get)
        
        response = {
            "model_used":           model_type,
            "routing":              "deep_learning",
            "predicted_pitch":      predicted_pitch,
            "confidence":           sorted_probs[predicted_pitch],
            "pitch_probabilities":  sorted_probs,
            "enrichment_latency_ms": enrichment_latency_ms,
            "enrichment_sources":   enrichment_sources,
        }

    # ------------------------------------------------------------------
    # auto 분기 — Per-Pitcher 지역 모델 또는 Stacking fallback 자동 선택
    # ------------------------------------------------------------------
    elif model_type == "auto":
        pitcher_id = input_data.pitcher
        local_model_path = LOCAL_DIR / f"{pitcher_id}.pkl"
        local_le_path    = LOCAL_DIR / f"{pitcher_id}_le.pkl"

        if local_model_path.exists() and local_le_path.exists():
            # -- Per-Pitcher 로컬 모델 사용
            cache_key = str(pitcher_id)
            if cache_key not in _per_pitcher_cache:
                local_m  = joblib.load(local_model_path)
                local_le = joblib.load(local_le_path)
                if hasattr(local_m, "set_params"):
                    try:
                        local_m.set_params(n_jobs=1)
                    except Exception:
                        pass
                _per_pitcher_cache[cache_key] = (local_m, local_le)

            local_m, local_le = _per_pitcher_cache[cache_key]

            # 투수별 피처 목록 추론
            if hasattr(local_m, "feature_names_in_"):
                feats = list(local_m.feature_names_in_)
            else:
                feats = list(local_m.get_booster().feature_names)

            X_local = _build_feature_vector(merged, feats)
            local_probs = local_m.predict_proba(X_local)[0]

            prob_dict = {
                str(local_le.classes_[i]): round(float(p), 4)
                for i, p in enumerate(local_probs)
            }
            sorted_probs    = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
            predicted_pitch = max(prob_dict, key=prob_dict.get)

            response = {
                "model_used":           "auto",
                "routing":              "per_pitcher",
                "pitcher_id":           pitcher_id,
                "predicted_pitch":      predicted_pitch,
                "confidence":           sorted_probs[predicted_pitch],
                "pitch_probabilities":  sorted_probs,
                "enrichment_latency_ms": enrichment_latency_ms,
                "enrichment_sources":   enrichment_sources,
            }
        else:
            # -- Scouting LLM Fallback (Before Stacking) --
            llm_probs = predict_with_scouting_llm(pitcher_id, enriched)
            if llm_probs is not None:
                sorted_probs = dict(sorted(llm_probs.items(), key=lambda x: x[1], reverse=True))
                predicted_pitch = max(llm_probs, key=llm_probs.get)
                
                response = {
                    "model_used":           "auto",
                    "routing":              "scouting_llm",
                    "pitcher_id":           pitcher_id,
                    "predicted_pitch":      predicted_pitch,
                    "confidence":           sorted_probs[predicted_pitch],
                    "pitch_probabilities":  sorted_probs,
                    "enrichment_latency_ms": enrichment_latency_ms,
                    "enrichment_sources":   enrichment_sources,
                }
            else:
                # -- Stacking fallback
                model_type = "stacking"
                _auto_fallback_routing = "stacking_fallback"
                
                # 아래 스태킹 코드로 넘겨서 실행하기 위한 로직
                meta_path = MODEL_DIR / 'stacking_meta_learner.pkl'
                paths_path = MODEL_DIR / 'stacking_base_model_paths.pkl'
    
                if not meta_path.exists() or not paths_path.exists():
                    raise HTTPException(
                        status_code=503,
                        detail="스태킹 메타러너 모델 또는 경로 정의 파일이 존재하지 않습니다."
                    )

                if not _stacking_cache:
                    base_paths = joblib.load(paths_path)
                    _stacking_cache["meta"]     = joblib.load(meta_path)
                    _stacking_cache["xgb"]      = joblib.load(MODEL_DIR / Path(base_paths['xgb']).name)
                    _stacking_cache["lgb"]      = joblib.load(MODEL_DIR / Path(base_paths['lgb']).name)
                    _stacking_cache["cat"]      = joblib.load(MODEL_DIR / Path(base_paths['cat']).name)
    
                meta_learner = _stacking_cache["meta"]
                best_xgb     = _stacking_cache["xgb"]
                best_lgb     = _stacking_cache["lgb"]
                best_cat     = _stacking_cache["cat"]

                for _m in (best_xgb, best_lgb, best_cat, meta_learner):
                    if hasattr(_m, "n_jobs"):
                        _m.n_jobs = 1
                    if hasattr(_m, "set_params"):
                        try: _m.set_params(n_jobs=1)
                        except Exception: pass
    
                xgb_feats = list(best_xgb.feature_names_) if hasattr(best_xgb, 'feature_names_') else list(best_xgb.feature_names_in_)
                lgb_feats = list(best_lgb.feature_names_) if hasattr(best_lgb, 'feature_names_') else list(best_lgb.feature_names_in_)
                cat_feats = list(best_cat.feature_names_) if hasattr(best_cat, 'feature_names_') else list(best_cat.feature_names_in_)
                
                X_xgb = _build_feature_vector(merged, xgb_feats)
                X_lgb = _build_feature_vector(merged, lgb_feats)
                X_cat = _build_feature_vector(merged, cat_feats)
            
                prob_xgb = best_xgb.predict_proba(X_xgb)[0]
                prob_lgb = best_lgb.predict_proba(X_lgb)[0]
                prob_cat = best_cat.predict_proba(X_cat)[0]
                
                X_meta = np.hstack([prob_xgb, prob_lgb, prob_cat]).reshape(1, -1)
                meta_probs = meta_learner.predict_proba(X_meta)[0]
                label_encoder = _load_label_encoder()
                pitch_classes = label_encoder.classes_
                
                prob_dict = {
                    str(pitch_classes[i]): round(float(p), 4)
                    for i, p in enumerate(meta_probs)
                }
                sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
                predicted_pitch = max(prob_dict, key=prob_dict.get)
                
                response = {
                    "model_used":           "auto",
                    "routing":              _auto_fallback_routing,
                    "predicted_pitch":      predicted_pitch,
                    "confidence":           sorted_probs[predicted_pitch],
                    "pitch_probabilities":  sorted_probs,
                    "enrichment_latency_ms": enrichment_latency_ms,
                    "enrichment_sources":   enrichment_sources,
                }

    # ------------------------------------------------------------------
    # 스태킹 분기 — XGBoost + LightGBM + CatBoost Level-1 Stacking
    # ------------------------------------------------------------------
    if model_type == "stacking" and 'response' not in locals():
        if not _stacking_cache:
            meta_path = MODEL_DIR / 'stacking_meta_learner.pkl'
            paths_path = MODEL_DIR / 'stacking_base_model_paths.pkl'

            if not meta_path.exists() or not paths_path.exists():
                raise HTTPException(
                    status_code=503,
                    detail="스태킹 메타러너 모델 또는 경로 정의 파일이 존재하지 않습니다."
                )

            base_paths = joblib.load(paths_path)
            _stacking_cache["meta"]     = joblib.load(meta_path)
            _stacking_cache["xgb"]      = joblib.load(MODEL_DIR / Path(base_paths['xgb']).name)
            _stacking_cache["lgb"]      = joblib.load(MODEL_DIR / Path(base_paths['lgb']).name)
            _stacking_cache["cat"]      = joblib.load(MODEL_DIR / Path(base_paths['cat']).name)

        meta_learner = _stacking_cache["meta"]
        best_xgb     = _stacking_cache["xgb"]
        best_lgb     = _stacking_cache["lgb"]
        best_cat     = _stacking_cache["cat"]

        # macOS ARM64 OMP 충돌 방지: 예측 직전 n_jobs 강제 단일 스레드
        for _m in (best_xgb, best_lgb, best_cat, meta_learner):
            if hasattr(_m, "n_jobs"):
                _m.n_jobs = 1
            if hasattr(_m, "set_params"):
                try:
                    _m.set_params(n_jobs=1)
                except Exception:
                    pass

        # 피처 벡터 구성 및 예측
        xgb_feats = list(best_xgb.feature_names_) if hasattr(best_xgb, 'feature_names_') else list(best_xgb.feature_names_in_)
        lgb_feats = list(best_lgb.feature_names_) if hasattr(best_lgb, 'feature_names_') else list(best_lgb.feature_names_in_)
        cat_feats = list(best_cat.feature_names_) if hasattr(best_cat, 'feature_names_') else list(best_cat.feature_names_in_)
        
        X_xgb = _build_feature_vector(merged, xgb_feats)
        X_lgb = _build_feature_vector(merged, lgb_feats)
        X_cat = _build_feature_vector(merged, cat_feats)
        
        prob_xgb = best_xgb.predict_proba(X_xgb)[0]
        prob_lgb = best_lgb.predict_proba(X_lgb)[0]
        prob_cat = best_cat.predict_proba(X_cat)[0]
        
        X_meta = np.hstack([prob_xgb, prob_lgb, prob_cat]).reshape(1, -1)
        
        meta_probs = meta_learner.predict_proba(X_meta)[0]
        label_encoder = _load_label_encoder()
        pitch_classes = label_encoder.classes_
        
        prob_dict = {
            str(pitch_classes[i]): round(float(p), 4)
            for i, p in enumerate(meta_probs)
        }
        
        sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
        predicted_pitch = max(prob_dict, key=prob_dict.get)
        
        response = {
            "model_used":           "stacking",
            "routing":              "stacking",
            "predicted_pitch":      predicted_pitch,
            "confidence":           sorted_probs[predicted_pitch],
            "pitch_probabilities":  sorted_probs,
            "enrichment_latency_ms": enrichment_latency_ms,
            "enrichment_sources":   enrichment_sources,
        }

    # ------------------------------------------------------------------
    # 앙상블 분기 — [Bypass Hotfix] Stacking 모델로 자동 우회
    # ------------------------------------------------------------------
    if model_type == "ensemble":
        model_type = "xgboost"

    # --- 단일 모델 분기 (xgboost / random_forest / lightgbm) ---
    if 'response' not in locals():
        model         = _load_model(model_type)
        label_encoder = _load_label_encoder()
        feature_names = list(model.feature_names_) if hasattr(model, 'feature_names_') else list(model.feature_names_in_)

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

        response = {
            "model_used":           model_type,
            "routing":              "single_model",
            "predicted_pitch":      predicted_pitch,
            "confidence":           sorted_probs[predicted_pitch],
            "pitch_probabilities":  sorted_probs,
            "enrichment_latency_ms": enrichment_latency_ms,
            "enrichment_sources":   enrichment_sources,
        }

    # --- [진단로그 추가] 라우팅 분석 및 클래스 진단 ---
    label_encoder = locals().get('label_encoder')
    if label_encoder is None:
        label_encoder = locals().get('local_le')
    if label_encoder is None:
        label_encoder = _load_label_encoder()
        
    proba = locals().get('probabilities')
    if proba is None:
        proba = locals().get('local_probs')
    if proba is None:
        proba = locals().get('meta_probs')
        
    proba_list = []
    if proba is not None:
        if hasattr(proba, "tolist"):
            proba_list = proba.tolist()
        else:
            proba_list = list(proba)

    print(f"[PREDICT] pitcher_id={input_data.pitcher}")
    print(f"[PREDICT] routing={response['routing']}")
    print(f"[PREDICT] model_used={response['model_used']}")
    print(f"[PREDICT] label_classes={label_encoder.classes_.tolist()}")
    if len(proba_list) > 0:
        print(f"[PREDICT] top3={sorted(zip(label_encoder.classes_, proba_list), key=lambda x: -x[1])[:3]}")
    else:
        print(f"[PREDICT] top3=No Probabilities Available")

    return response


@router.post(
    "/cache/rebuild",
    summary="enrichment pkl 캐시 재빌드",
    tags=["Cache"],
)
def rebuild_cache():
    """
    [캐시 재빌드 엔드포인트]
    - build_cache.py의 build_enrichment_cache()를 서버 내에서 직접 호출
    - n8n post_game 워크플로우에서 매일 새벽 자동 호출
    - 수동 갱신 시에도 사용 가능
    """
    try:
        from ml_engine.build_cache import build_enrichment_cache
        build_enrichment_cache()

        # 메모리 캐시 초기화 (다음 요청 시 새 pkl 로드)
        from backend.services.enrichment import _cache
        _cache.clear()

        return {
            "status": "success",
            "message": "enrichment pkl 캐시 재빌드 완료",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"캐시 재빌드 실패: {str(e)}")