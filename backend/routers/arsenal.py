from fastapi import APIRouter, Query, HTTPException
from backend.services.arsenal_cache import get_pitcher_arsenal

router = APIRouter()

@router.get("/pitcher-arsenal")
def get_pitcher_arsenal_endpoint(
    pitcherId: int = Query(..., description="조회하고자 하는 투수의 ID (필수)"),
    year: int = Query(2024, description="조회 데이터의 기준 연도 (기본값 2024)")
):
    """
    [특정 투수의 구종 레퍼토리 및 평균 구속/좌표 시각화 데이터 조회 API]
    """
    arsenal = get_pitcher_arsenal(pitcherId, year)
    if arsenal is None:
        raise HTTPException(status_code=404, detail="pitcher not found")
    
    # fallback 등의 로직이 arsenal_cache.py 내부에서 수행되었을 수 있으므로
    # 실제 반환된 arsenal의 연도가 다를 수 있지만, 엔드포인트 응답에는 요청한 연도를 일치시키거나 혹은 식별용으로 내려줌.
    # 안전하게 요청 연도로 응답에 세팅
    return {
        "pitcher_id": pitcherId,
        "year": year,
        "arsenal": arsenal
    }
