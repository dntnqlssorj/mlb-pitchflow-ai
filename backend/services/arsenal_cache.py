import logging
import joblib
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_PATH = Path("ml_engine/cache/pitcher_arsenal_cache.pkl")

# 모듈 레벨 싱글톤 변수
_cache: pd.DataFrame | None = None

PITCH_NAME_MAP = {
    "FF": "포심 패스트볼", "SI": "싱커", "FC": "커터", "FT": "투심 패스트볼",
    "SL": "슬라이더", "ST": "스위퍼", "SV": "슬러브", "CU": "커브", "KC": "너클커브",
    "CH": "체인지업", "FS": "스플리터", "FO": "포크볼", "KN": "너클볼",
    "SC": "스크루볼", "EP": "이파수", "CS": "슬로우커브", "FA": "패스트볼"
}

PITCH_COLOR_MAP = {
    "FF": "#EF4444", "SI": "#F97316", "FC": "#EAB308", "FT": "#F59E0B",
    "SL": "#22C55E", "ST": "#10B981", "SV": "#14B8A6", "CU": "#3B82F6",
    "KC": "#6366F1", "CH": "#EC4899", "FS": "#8B5CF6", "FO": "#A855F7",
    "KN": "#06B6D4", "SC": "#84CC16", "EP": "#F43F5E", "CS": "#0EA5E9", "FA": "#FB923C"
}

def _load_cache() -> pd.DataFrame:
    """
    [pitcher_arsenal_cache.pkl 싱글톤 로더]
    """
    global _cache
    if _cache is None:
        if not CACHE_PATH.exists():
            logger.error(f"구종 레퍼토리 캐시 파일 없음: {CACHE_PATH} — build_cache.py 실행이 선행되어야 합니다.")
            _cache = pd.DataFrame(columns=['pitcher', 'game_year', 'pitch_type', 'pct', 'avg_speed', 'avg_plate_x', 'avg_plate_z'])
        else:
            try:
                _cache = joblib.load(CACHE_PATH)
                logger.info(f"구종 레퍼토리 캐시 로드 성공 (총 {len(_cache)}개 레코드)")
            except Exception as e:
                logger.error(f"구종 레퍼토리 캐시 로드 실패: {e}")
                _cache = pd.DataFrame(columns=['pitcher', 'game_year', 'pitch_type', 'pct', 'avg_speed', 'avg_plate_x', 'avg_plate_z'])
    return _cache

def get_pitcher_arsenal(pitcher_id: int, year: int) -> list[dict] | None:
    """
    [특정 투수의 연도별 구종 레퍼토리 리스트 조회]
    반환 딕셔너리 필드: pitch_type, name, pct, avg_speed, avg_plate_x, avg_plate_z, color
    정렬: pct 내림차순
    """
    df = _load_cache()
    if df.empty:
        return None

    # 임시 로그 추가
    print(f"[ARSENAL] querying pitcher_id={pitcher_id} type={type(pitcher_id)}")
    print(f"[ARSENAL] cache pitcher col dtype={df['pitcher'].dtype}")

    # 투수 ID 필터링 (조회 전 타입 강제 통일)
    pitcher_id = int(pitcher_id)
    p_df = df[df['pitcher'].astype(int) == pitcher_id]
    
    print(f"[ARSENAL] matched rows={len(p_df)}")
    if len(p_df) > 0:
        print(f"[ARSENAL] first row={p_df.iloc[0].to_dict()}")

    if p_df.empty:
        logger.info(f"투수 ID {pitcher_id}에 해당하는 데이터가 캐시에 없습니다.")
        return None

    available_years = [int(y) for y in p_df['game_year'].unique() if pd.notna(y)]
    target_year = int(year)

    # 연도(year) 존재 여부 체크 및 fallback 처리
    if target_year not in available_years:
        if len(available_years) > 0:
            target_year = int(max(available_years))
            logger.warning(
                f"투수 ID {pitcher_id}에 대해 요청된 연도 {year} 데이터가 없어 "
                f"가장 최근 연도인 {target_year} 데이터로 Fallback 합니다."
            )
        else:
            return None

    # 최종 대상 연도 필터링
    year_df = p_df[p_df['game_year'].astype(int) == target_year]
    if year_df.empty:
        return []

    # pct 내림차순 정렬
    sorted_df = year_df.sort_values(by='pct', ascending=False)

    arsenal_list = []
    for _, row in sorted_df.iterrows():
        pt = str(row['pitch_type'])
        avg_speed = row['avg_speed']
        avg_plate_x = row['avg_plate_x']
        avg_plate_z = row['avg_plate_z']
        
        # NaN 값 처리 (float | None 변환)
        avg_speed_val = float(avg_speed) if pd.notna(avg_speed) else None
        avg_plate_x_val = float(avg_plate_x) if pd.notna(avg_plate_x) else None
        avg_plate_z_val = float(avg_plate_z) if pd.notna(avg_plate_z) else None

        arsenal_list.append({
            "pitch_type": pt,
            "name": PITCH_NAME_MAP.get(pt, pt),
            "pct": round(float(row['pct']), 2),
            "avg_speed": avg_speed_val,
            "avg_plate_x": avg_plate_x_val,
            "avg_plate_z": avg_plate_z_val,
            "color": PITCH_COLOR_MAP.get(pt, "#888888")
        })

    return arsenal_list
