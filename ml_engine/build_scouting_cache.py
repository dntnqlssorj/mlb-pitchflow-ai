import pandas as pd
import requests
import joblib
import time
import logging
import os
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("ml_engine/cache")
MODEL_DIR = Path("ml_engine/models")

def get_mlb_player_id(name: str) -> int:
    """MLB Stats API를 호출하여 선수의 player_id를 반환 (rate limit 방지 0.2초 슬립)"""
    url = f"https://statsapi.mlb.com/api/v1/people/search?names={name}"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            people = data.get("people", [])
            if people:
                # 첫 번째 매칭 반환
                time.sleep(0.2)
                return people[0].get("id")
    except Exception as e:
        logger.warning(f"MLB API 검색 실패 ({name}): {e}")
    
    time.sleep(0.2)
    return None

def parse_grade(grade_str: str) -> int:
    """'55/60' 같은 문자열에서 현재 등급(55) 추출"""
    if pd.isna(grade_str) or not isinstance(grade_str, str):
        return 0
    match = re.match(r"^(\d+)", str(grade_str).strip())
    if match:
        return int(match.group(1))
    return 0

def map_fb_type(fb_type_str: str) -> str:
    """FB Type 문자열을 FF, SI, FC로 변환"""
    if pd.isna(fb_type_str):
        return "FF"
    val = str(fb_type_str).lower()
    if "cut" in val:
        return "FC"
    if "sink" in val or "tail" in val:
        return "SI"
    if "rise" in val or "downhill" in val or "uphill" in val:
        return "FF"
    return "FF"  # Default

def build_scouting_cache():
    logger.info("Scouting 데이터 파싱 및 캐시 빌드 시작...")
    
    summary_path = "data/scouting/Summary.csv"
    pitching_path = "data/scouting/Scouting-pitching.csv"
    
    if not os.path.exists(summary_path) or not os.path.exists(pitching_path):
        logger.error("Scouting CSV 파일이 존재하지 않습니다.")
        return

    df_sum = pd.read_csv(summary_path)
    df_pit = pd.read_csv(pitching_path, skiprows=1)

    # Summary 필터링: Current Level == "MLB" AND Pos in (SP, SIRP, MIRP, RP)
    valid_pos = ["SP", "SIRP", "MIRP", "RP"]
    df_sum_filtered = df_sum[
        (df_sum["Current Level"] == "MLB") & 
        (df_sum["Pos"].isin(valid_pos))
    ].copy()

    # Name 기준으로 병합
    df_merged = pd.merge(df_sum_filtered, df_pit, on="Name", how="inner")
    
    logger.info(f"병합된 대상 투수: {len(df_merged)}명. MLB Stats API 매핑 시작...")

    cache_data = {}
    found_count = 0

    for idx, row in df_merged.iterrows():
        name = row["Name"]
        player_id = get_mlb_player_id(name)
        
        if not player_id:
            logger.warning(f"Player ID 매핑 실패: {name}")
            continue
            
        found_count += 1
        
        # 1. 등급 추출
        fb_grade = parse_grade(row.get("FB"))
        sl_grade = parse_grade(row.get("SL"))
        cb_grade = parse_grade(row.get("CB"))
        ch_grade = parse_grade(row.get("CH"))
        
        # 2. 비율 변환
        total_grade = fb_grade + sl_grade + cb_grade + ch_grade
        if total_grade == 0:
            logger.warning(f"{name} (ID: {player_id}) 구종 등급 정보 없음. 스킵합니다.")
            continue
            
        pct_fb = fb_grade / total_grade
        pct_sl = sl_grade / total_grade
        pct_cb = cb_grade / total_grade
        pct_ch = ch_grade / total_grade
        
        # 3. FB Type 기반 패스트볼 매핑
        fb_type = map_fb_type(row.get("FB Type"))
        
        # 결과 딕셔너리 생성 (18개 클래스 포맷에 맞춤)
        # 스카우팅 리포트에 없는 구종은 0.0으로 둠
        pitch_probs = {
            "FF": pct_fb if fb_type == "FF" else 0.0,
            "SI": pct_fb if fb_type == "SI" else 0.0,
            "FC": pct_fb if fb_type == "FC" else 0.0,
            "SL": pct_sl,
            "CU": pct_cb, # CB는 커브(CU)로 매핑
            "CH": pct_ch,
        }
        
        cache_data[player_id] = {
            "name": name,
            "grades": {
                "FB": fb_grade,
                "SL": sl_grade,
                "CB": cb_grade,
                "CH": ch_grade
            },
            "fb_type": fb_type,
            "base_probs": pitch_probs
        }
        
        if found_count % 10 == 0:
            logger.info(f"API 매핑 진행 중... ({found_count}/{len(df_merged)})")

    # 캐시 파일 저장
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "scouting_cache.pkl"
    joblib.dump(cache_data, cache_path, compress=3)
    
    logger.info(f"[성공] scouting_cache.pkl 저장 완료! (총 {len(cache_data)}명)")

if __name__ == "__main__":
    build_scouting_cache()
