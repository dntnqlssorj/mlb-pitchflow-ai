import os
# - macOS Uvicorn Fork Reload 환경에서 OpenBLAS / OMP 스레딩 데드락 방지 은탄환
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_INIT_AT_FORK"] = "FALSE"
# - XGBoost / LightGBM 단일 스레드 강제 (macOS ARM64 OMP 충돌 수정)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["LIGHTGBM_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


import logging
import traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from starlette.exceptions import HTTPException as StarletteHTTPException
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


logger = logging.getLogger("pitchflow")


@app.exception_handler(StarletteHTTPException)
@app.exception_handler(FastAPIHTTPException)
async def http_exception_handler(request: Request, exc):
    """
    [FastAPI HTTPException 핸들러]
    - 목적: 명시적으로 raise된 HTTPException을 구조화된 JSON으로 반환
    - 클라이언트가 status / detail 필드로 에러 원인을 파악 가능
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "code":   exc.status_code,
            "detail": exc.detail,
        },
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    [전역 예외 핸들러 — 런타임 에러 방어망]
    - 목적: try/except로 잡히지 않은 모든 예외를 500 JSON으로 안전하게 래핑
    - 보안: 스택 트레이스는 서버 로그에만 기록, 응답에는 에러 유형만 노출
    - 대상: ValueError (피처 차원 불일치), KeyError (pkl 키 오류), torch 런타임 오류 등
    """
    error_type = type(exc).__name__
    logger.error(
        f"[GlobalExceptionHandler] {error_type}: {exc}\n"
        f"Request: {request.method} {request.url}\n"
        f"{traceback.format_exc()}"
    )
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "code":   500,
            "detail": f"서버 내부 오류 ({error_type}). 관리자에게 문의하세요.",
        },
    )


@app.get("/", tags=["Health"])
def health_check():
    """
    [헬스 체크 엔드포인트]
    - 목적: 서버 정상 작동 여부 확인
    """
    return {
        "status": "🟢 MLB PitchFlow AI 서버 정상 작동 중",
        "docs": "http://localhost:8000/docs",
        "available_models": ["xgboost", "random_forest", "lightgbm", "catboost", "stacking", "bilstm", "transformer", "auto"]
    }
