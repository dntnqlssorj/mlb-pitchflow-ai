import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional

router = APIRouter()

# - 모델 디렉토리: ml_engine/models/ 기준 경로 고정
MODEL_DIR = Path("ml_engine/models")

# - 모델명 → pkl 파일명 매핑 테이블 (새 모델 추가 시 이 딕셔너리만 확장)
MODEL_MAP = {
    "xgboost": "xgboost_pitch_model.pkl",
    "random_forest": "random_forest_pitch_model.pkl",
    "lightgbm": "lightgbm_pitch_model.pkl",
}

class PitchInferenceInput(BaseModel):
    """
    [추론 요청 데이터 스키마 (Pydantic)]
    - 목적: n8n/백엔드가 전송하는 경기 상황 피처 타입 검증 및 문서화
    - 설계: 핵심 도메인 피처만 필수 입력, 나머지는 모델 학습 피처 목록 기반 0 패딩 자동 처리
    """
    # - 투수 식별 및 체력 관련
    pitcher: int = Field(..., description="투수 MLB player_id")
    pitch_count_in_game: int = Field(..., description="해당 경기 누적 투구 수")
    stamina_index: float = Field(0.0, description="투수 체력 저하 지수 (0.0 = 최상)")
    velocity_decay_ratio: float = Field(1.0, description="현재 구속 / 기준 구속 비율")
    spin_decay_ratio: float = Field(1.0, description="현재 회전수 / 기준 회전수 비율")
    base_speed: float = Field(0.0, description="해당 경기 기준 초반 구속 (mph)")
    base_spin: float = Field(0.0, description="해당 경기 기준 초반 회전수 (rpm)")
    
    # - 타자 식별
    batter: Optional[int] = Field(None, description="타자 MLB player_id")
    
    # - 포수 및 위기 상황
    fielder_2: Optional[int] = Field(None, description="포수 MLB player_id")
    is_crisis: int = Field(0, description="위기 상황 여부 (3루 주자 존재 시 1)")
    blocking_leverage_factor: float = Field(0.0, description="포수 블로킹 레버리지 팩터")
    catcher_blocking_runs: float = Field(0.0, description="포수 블로킹 런 수치")
    
    # - 수비진 OAA
    team_oaa_total: float = Field(0.0, description="당일 출전 야수진 OAA 합산")
    fielding_risk_index: float = Field(0.0, description="수비 리스크 인덱스")
    
    # - 볼카운트 및 경기 상황
    balls: int = Field(0, description="볼 카운트")
    strikes: int = Field(0, description="스트라이크 카운트")
    outs_when_up: int = Field(0, description="현재 아웃 카운트")
    inning: int = Field(1, description="이닝")
    game_year: int = Field(2025, description="시즌 연도")
    on_1b: int = Field(0, description="1루 주자 여부 (있으면 player_id, 없으면 0)")
    on_2b: int = Field(0, description="2루 주자 여부")
    on_3b: int = Field(0, description="3루 주자 여부")
    
    # - 현재 투구 물리 데이터 (선택)
    release_speed: Optional[float] = Field(None, description="투구 구속 (mph)")
    release_spin_rate: Optional[int] = Field(None, description="투구 회전수 (rpm)")
    spin_axis: Optional[int] = Field(None, description="스핀 축")
    release_pos_x: Optional[float] = Field(None, description="릴리즈 포인트 X")
    release_pos_z: Optional[float] = Field(None, description="릴리즈 포인트 Z")
    release_pos_y: Optional[float] = Field(None, description="릴리즈 포인트 Y")
    release_extension: Optional[float] = Field(None, description="릴리즈 익스텐션")
    arm_angle: Optional[float] = Field(None, description="투구 팔 각도")
    effective_speed: Optional[float] = Field(None, description="체감 구속")
    api_break_z_with_gravity: Optional[float] = Field(None, description="수직 무브먼트 (중력 포함)")
    api_break_x_arm: Optional[float] = Field(None, description="수평 무브먼트 (투구 팔 기준)")
    api_break_x_batter_in: Optional[float] = Field(None, description="수평 무브먼트 (타자 기준)")
    pfx_x: Optional[float] = Field(None, description="피칭 fx (수평)")
    pfx_z: Optional[float] = Field(None, description="피칭 fz (수직)")
    plate_x: Optional[float] = Field(None, description="홈플레이트 X 좌표")
    plate_z: Optional[float] = Field(None, description="홈플레이트 Z 좌표")
    
    # - 투수 투구 이력 (선택)
    n_thruorder_pitcher: Optional[int] = Field(None, description="타순 순환 횟수")
    pitcher_days_since_prev_game: Optional[int] = Field(None, description="직전 등판 후 경과일")

def _load_model(model_type: str):
    """
    [동적 모델 로드]
    - 목적: model_type 파라미터 기반 pkl 파일 동적 매핑 및 로드
    - 예외 처리: 미존재 모델 요청 시 HTTP 404 반환
    """
    # - 모델명 유효성 검사: MODEL_MAP 미등록 모델 요청 차단
    if model_type not in MODEL_MAP:
        available = list(MODEL_MAP.keys())
        raise HTTPException(
            status_code=404,
            detail=f"모델 '{model_type}'을 찾을 수 없습니다. 사용 가능한 모델: {available}"
        )
    
    # - 파일 존재 여부 확인: 실제 pkl 파일 미생성 시 503 반환
    model_path = MODEL_DIR / MODEL_MAP[model_type]
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"모델 파일이 아직 학습되지 않았습니다: {model_path}"
        )
    
    # - 모델 로드: joblib 역직렬화
    return joblib.load(model_path)

def _load_label_encoder():
    """
    [라벨 인코더 로드]
    - 목적: 숫자형 예측값 → 구종 문자열 역변환용 인코더 로드
    """
    encoder_path = MODEL_DIR / "label_encoder.pkl"
    if not encoder_path.exists():
        raise HTTPException(status_code=503, detail="라벨 인코더 파일이 존재하지 않습니다.")
    return joblib.load(encoder_path)

def _build_feature_vector(input_data: PitchInferenceInput, feature_names: list) -> pd.DataFrame:
    """
    [피처 벡터 구성]
    - 목적: 모델 학습 피처 목록 기반 자동 정렬 및 결측 피처 0 패딩
    - 방법: 입력 데이터를 딕셔너리 → DataFrame으로 변환 후 학습 피처 목록에 맞게 재정렬
    """
    # - 입력 데이터를 딕셔너리로 변환 (None → 0 대체)
    input_dict = {
        k: (v if v is not None else 0)
        for k, v in input_data.model_dump().items()
    }
    
    # - 단일 행 DataFrame 생성
    df = pd.DataFrame([input_dict])
    
    # - 학습 피처 목록 기준 재정렬: 없는 컬럼은 0으로 자동 패딩
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    return df[feature_names].fillna(0)

@router.post(
    "/pitch",
    summary="다음 투구 구종 예측",
    description="현재 경기 상황 피처를 입력받아 다음 투구 구종 확률 분포를 반환합니다.",
    tags=["Prediction"]
)
def predict_pitch(
    input_data: PitchInferenceInput,
    model_type: str = Query(default="xgboost", description="사용할 모델 (xgboost / random_forest / lightgbm)")
):
    """
    [구종 예측 엔드포인트]
    - 목적: 실시간 경기 상황 → 다음 투구 구종 확률 JSON 반환
    - 흐름: 모델 로드 → 피처 벡터 자동 정렬 → 예측 → 라벨 디코딩 → 확률 응답
    """
    # - 모델 및 인코더 로드: model_type 파라미터 기반 동적 로드
    model = _load_model(model_type)
    label_encoder = _load_label_encoder()
    
    # - 학습 피처 목록 추출: 모델 객체에서 직접 읽어 자동 정렬 기준으로 활용
    feature_names = list(model.feature_names_in_)
    
    # - 피처 벡터 구성: 입력 → 학습 피처 정렬 → 결측 컬럼 0 패딩
    X = _build_feature_vector(input_data, feature_names)
    
    # - 클래스별 확률 예측: predict_proba로 전체 구종 분포 산출
    probabilities = model.predict_proba(X)[0]
    
    # - 라벨 디코딩: 숫자 인덱스를 구종 문자열(FF, SL 등)로 역변환
    pitch_classes = label_encoder.classes_
    prob_dict = {
        str(pitch_classes[i]): round(float(prob), 4)
        for i, prob in enumerate(probabilities)
    }
    
    # - 우선순위 정렬: 확률 높은 순으로 응답 구성
    sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
    predicted_pitch = max(prob_dict, key=prob_dict.get)
    
    return {
        "model_used": model_type,
        "predicted_pitch": predicted_pitch,
        "confidence": sorted_probs[predicted_pitch],
        "pitch_probabilities": sorted_probs
    }
