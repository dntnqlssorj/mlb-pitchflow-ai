from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import predict

# - 앱 인스턴스 생성: 메타데이터 및 Swagger 문서 제목 설정
app = FastAPI(
    title="MLB PitchFlow AI - 구종 예측 API",
    description="""
    ## ⚾️ MLB PitchFlow AI 추론 서버
    
    n8n 파이프라인 및 프론트엔드와 연동하여 **실시간 투구 구종 예측**을 수행하는 API 서버입니다.
    
    ### 주요 기능
    - **멀티 모델 스위칭**: `model_type` 파라미터로 XGBoost / LightGBM / RandomForest 전환
    - **구종 확률 분포 반환**: 패스트볼, 슬라이더, 체인지업 등 전체 구종 확률 JSON 응답
    - **도메인 피처 반영**: 투수 체력(Stamina), 포수 블로킹(Leverage), 수비 OAA(Risk Index) 적용
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# - CORS 미들웨어: n8n 및 프론트엔드 크로스 오리진 요청 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# - 라우터 등록: 예측 엔드포인트 /predict 접두사로 마운트
app.include_router(predict.router, prefix="/predict")

@app.get("/", tags=["Health"])
def health_check():
    """
    [헬스 체크 엔드포인트]
    - 목적: 서버 정상 작동 여부 확인
    """
    return {
        "status": "🟢 MLB PitchFlow AI 서버 정상 작동 중",
        "docs": "http://localhost:8000/docs",
        "available_models": ["xgboost", "random_forest", "lightgbm"]
    }
