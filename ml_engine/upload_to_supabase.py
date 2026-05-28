import os
import sys
import time
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

from ml_engine.datasets import get_clean_datasets
from ml_engine.feature_engineering import (
    build_season_baseline,
    calculate_pitcher_stamina_decay,
    integrate_catcher_blocking,
    integrate_fielding_oaa,
    add_situational_features,
    add_pitch_sequence_features,
    add_pitcher_repertoire_features,
    add_pitcher_situation_features,
)
from ml_engine.config import ALLOWED_FEATURES

# - 환경 변수 로드: .env 파일 자동 감지 및 로드
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # - 서비스 롤 키(Service Role Key) 권장

def get_supabase_client() -> Client:
    """
    [Supabase 클라이언트 객체 초기화]
    - 목적: Supabase 서비스 연결 및 클라이언트 인스턴스 반환
    """
    # - 환경 변수 검증: 인증 정보 누락 시 실행 차단 및 안내
    if not SUPABASE_URL or not SUPABASE_KEY or "your_" in SUPABASE_URL:
        print("\n⚠️ [경고] .env 파일에 올바른 SUPABASE_URL 및 SUPABASE_KEY를 설정해 주세요.")
        print("💡 템플릿 예시:")
        print("   SUPABASE_URL=https://your-project.supabase.co")
        print("   SUPABASE_KEY=your-service-role-key\n")
        
        # - .env 템플릿 파일 자동 생성
        if not os.path.exists(".env"):
            with open(".env", "w", encoding="utf-8") as f:
                f.write("SUPABASE_URL=https://your-project-id.supabase.co\n")
                f.write("SUPABASE_KEY=your-service-role-key\n")
            print("📝 프로젝트 루트에 '.env' 템플릿 파일을 생성했습니다. 정보를 입력해 주세요.")
            
        sys.exit(1)
        
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def prepare_master_dataset() -> pd.DataFrame:
    """
    [마스터 데이터프레임 빌드]
    - 목적: 4개 원본 데이터셋 정제 및 3대 도메인 피처 결합 후 ALLOWED_FEATURES 38개 컬럼만 반환
    """
    print("📦 원본 데이터 로드 및 피처 가공 시작...")
    
    # - 데이터 로드: datasets.py 연동
    datasets = get_clean_datasets()
    bat_df = datasets['bat_tracking']
    
    # - 도메인 피처 통합: feature_engineering.py 연동
    # - 샘플링 없이 전체 데이터 기준으로 시즌 평균 구속/회전수 사전 산출 (누수 방지)
    season_baseline_df = build_season_baseline(bat_df)
    df = calculate_pitcher_stamina_decay(bat_df, season_baseline_df, baseline_pitches=15)
    df = integrate_catcher_blocking(df, datasets['blocking'])
    df = integrate_fielding_oaa(df, datasets['oaa'])
    df = add_situational_features(df)
    df = add_pitch_sequence_features(df)
    df = add_pitcher_repertoire_features(df)
    df = add_pitcher_situation_features(df)
    
    # - [수정 3] NaN/inf/-inf → None 변환 (replace 방식)
    # - apply()로 None 반환해도 to_dict() 시 pandas가 numpy float64 nan으로 되돌려버림
    # - replace()는 to_dict() 후에도 None이 유지됨
    df = df.replace([np.inf, -np.inf], None)
    df = df.where(pd.notnull(df), None)

    # - [수정 4] BIGINT 컬럼 타입 강제 변환 (3단계) — ALLOWED_FEATURES 내 정수 컬럼만 대상
    # - 피처 엔지니어링 후 정수 컬럼들이 float64로 변환됨 (예: is_risp = 1.0)
    # - DB BIGINT에 "1.0" 입력 시 "invalid input syntax for type bigint" 에러 발생
    # - 1단계: pd.to_numeric(errors="coerce") → float64 (불가값은 NaN)
    # - 2단계: .where(notna(), pd.NA)          → NaN을 Int64가 인식하는 pd.NA로 교체
    # - 3단계: .astype("Int64")               → nullable 정수 타입으로 최종 변환
    bigint_cols = [
        # 그룹 A: 경기 상황
        "balls", "strikes", "outs_when_up", "inning",
        "on_1b", "on_2b", "on_3b",
        "home_score_diff", "bat_score_diff",
        # 그룹 B: 투수 이력
        "pitcher", "game_year", "n_thruorder_pitcher",
        "pitcher_days_since_prev_game", "age_pit",
        # 그룹 C: 타자 이력
        "batter", "stand",
        "n_priorpa_thisgame_player_at_bat", "batter_days_since_prev_game", "age_bat",
        # 그룹 D/E: 체력·포수
        "pitch_count_in_game", "is_risp",
        # 그룹 E/F: 수비수 ID
        "fielder_2", "fielder_3", "fielder_4", "fielder_5",
        "fielder_6", "fielder_7", "fielder_8", "fielder_9",
        # 그룹 G: PK 식별자
        "game_pk", "at_bat_number", "pitch_number",
        # 신규 정수 타입 피처
        "count_situation", "matchup_type",
        "prev_pitch_1", "prev_pitch_2", "prev_pitch_3",
    ]
    for col in bigint_cols:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            df[col] = numeric.where(numeric.notna(), pd.NA).astype("Int64")

    # - ALLOWED_FEATURES 38개 컬럼만 선택 (나머지 91개 누수/불필요 컬럼 제거)
    upload_cols = [c for c in ALLOWED_FEATURES if c in df.columns]
    missing = [c for c in ALLOWED_FEATURES if c not in df.columns]
    if missing:
        print(f"⚠️ ALLOWED_FEATURES 중 데이터에 없는 컬럼: {missing}")
    df = df[upload_cols]

    print(f"✅ 마스터 데이터 가공 완료: {df.shape[0]} 행, {df.shape[1]} 개 컬럼")
    return df

def _safe_none(v):
    """
    [수정 6] to_dict() 후 딕셔너리 단계에서 모든 비직렬화 타입 강제 변환

    - 1단계: pd.isna() → NaN, None, pd.NA 처리
    - 2단계: inf/-inf → None (pd.isna가 False로 통과시킴)
    - 3단계: numpy float32/float16 → Python float
    - 4단계: numpy 정수 타입 → Python int
    - 5단계: numpy bool → Python bool
    - 6단계: datetime/Timestamp → 문자열
    """
    # 1단계: NaN, None, pd.NA 처리
    # pd.isna()는 NaN, None, pd.NA 세 가지 모두 True 반환
    # try/except: 문자열 등 일부 타입에서 TypeError 발생 가능하여 방어
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass

    # 2단계: inf/-inf 처리
    # pd.isna(float('inf')) = False → pd.isna()가 통과시키므로 별도 처리 필요
    if isinstance(v, float) and (v == float('inf') or v == float('-inf')):
        return None

    # 3단계: numpy float32/float16 → Python float
    # 원인: numpy float32는 json.dumps에서 "not JSON serializable" 에러 발생
    if isinstance(v, (np.float32, np.float16)):
        return float(v)

    # 4단계: numpy 정수 타입 → Python int
    # 원인: np.int64, np.int32 등은 json.dumps에서 "not JSON serializable" 에러 발생
    if isinstance(v, (np.int64, np.int32, np.int16, np.int8,
                      np.uint64, np.uint32, np.uint16, np.uint8)):
        return int(v)

    # 5단계: numpy bool → Python bool
    # 원인: np.bool_(True)는 json.dumps에서 "not JSON serializable" 에러 발생
    if isinstance(v, np.bool_):
        return bool(v)

    # 6단계: datetime/Timestamp → 문자열
    # 원인: Python datetime, pandas Timestamp는 json.dumps에서 직렬화 불가
    import datetime
    if isinstance(v, (datetime.date, datetime.datetime, pd.Timestamp)):
        return str(v)

    return v


def upload_in_batches(
    supabase: Client, 
    df: pd.DataFrame, 
    table_name: str = "statcast_bat_tracking", 
    chunk_size: int = 1000,
    pilot_mode: bool = True
):
    """
    [대용량 벌크 업로드 파이프라인]
    - 목적: ALLOWED_FEATURES 38개 컬럼 데이터를 메모리 에러와 속도 지연 없이 분할 적재
    - 방법: Chunk 단위 슬라이싱 및 Exponential Backoff 재시도 로직 적용
    """
    # - 테스트 모드 제어: pilot_mode 활성화 시 1개 청크(1,000행)만 적재하고 조기 종료
    if pilot_mode:
        print(f"\n🧪 [파일럿 테스트 모드 활성화] 상위 {chunk_size}개 행만 테스트 업로드합니다.")
        df_target = df.head(chunk_size).copy()
    else:
        print(f"\n🚀 [실전 전체 업로드 시작] 총 {len(df)}개 행을 {chunk_size}행 단위로 분할 적재합니다.")
        df_target = df.copy()
        
    total_rows = len(df_target)
    total_chunks = math.ceil(total_rows / chunk_size)
    
    print(f"📊 총 청크 개수: {total_chunks}개 (청크 크기: {chunk_size})")
    
    # - 순회 분할 업로드 시작
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)
        
        # - 데이터 슬라이싱 및 사전형(Dict) 리스트 변환
        chunk_df = df_target.iloc[start_idx:end_idx]
        chunk_data = chunk_df.to_dict(orient="records")
        # - [수정 6] to_dict() 후 NaN/inf/pd.NA → None 강제 변환
        chunk_data = [{k: _safe_none(v) for k, v in record.items()} for record in chunk_data]
        
        # - 재시도 로직 설정: 네트워크 차단 및 레이트 리밋 대비 최대 5회 재시도
        max_retries = 5
        retry_delay = 2
        success = False
        
        for attempt in range(max_retries):
            try:
                # - Supabase API 호출: upsert 실행 (PK 중복 시 덮어쓰기)
                # - [수정] insert → upsert 전환
                # - 전환 이유:
                #   insert는 PK 중복 시 에러 발생 → 재적재 시 중단됨
                #   upsert는 PK 중복 시 기존 행 업데이트 → 멱등성 확보
                #   멱등성: 몇 번 실행해도 같은 결과 보장
                #           중간에 끊겨도 다시 실행하면 이어서 처리 가능
                # - on_conflict: PK 컬럼(game_pk, at_bat_number, pitch_number) 기준
                #   충돌 감지 → 동일 투구 데이터 중복 적재 자동 방어
                supabase.table(table_name).upsert(
                    chunk_data,
                    on_conflict='game_pk,at_bat_number,pitch_number'
                ).execute()
                success = True
                break
            except Exception as e:
                print(f"\n⚠️ [{i+1}/{total_chunks} 청크] 업로드 실패 (시도 {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # - 지수 백오프 적용
                else:
                    print("❌ [오류] 재시도 횟수 초과로 업로드를 중단합니다.")
                    sys.exit(1)
                    
        # - 진행률 로깅: 실시간 업로드 현황 출력
        progress = ((i + 1) / total_chunks) * 100
        print(f" 🟩 [{i+1}/{total_chunks} 청크 완료] {start_idx} ~ {end_idx} 행 적재 성공 ({progress:.1f}%)", end="\r")
        sys.stdout.flush()
        
    print(f"\n\n🎉 [업로드 성공] 총 {total_rows}개의 데이터가 '{table_name}' 테이블에 완벽히 적재되었습니다!")

if __name__ == "__main__":
    # - 1. Supabase 연결 초기화
    supabase_client = get_supabase_client()
    
    # - 2. 마스터 데이터 준비
    master_df = prepare_master_dataset()
    
    # - 3. 전체 데이터 업로드 (144만 행 전체 적재 — enrichment 조회 정상화 목적)
    upload_in_batches(
        supabase=supabase_client,
        df=master_df,
        # - Supabase에 데이터를 올릴 대상 테이블 이름
        table_name="statcast_bat_tracking",
        # - 한 번에 올리는 행 수: 1,000행씩 나눠서 업로드 (메모리 초과 방지)
        chunk_size=1000,
        # - False: 144만 행 전체 업로드 (True였을 때는 1,000행만 테스트 업로드)
        # - 주의: Supabase 대시보드에서 기존 1,000행 삭제 후 실행할 것 (중복 에러 방지)
        pilot_mode=False
    )
