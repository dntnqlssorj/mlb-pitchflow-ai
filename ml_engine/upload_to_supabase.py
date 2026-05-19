import os
import sys
import time
import math
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

from ml_engine.datasets import get_clean_datasets
from ml_engine.feature_engineering import (
    calculate_pitcher_stamina_decay,
    integrate_catcher_blocking,
    integrate_fielding_oaa
)

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
    - 목적: 4개 원본 데이터셋 정제 및 3대 도메인 피처 결합 완료된 최종 129개 컬럼 데이터셋 반환
    """
    print("📦 원본 데이터 로드 및 피처 가공 시작...")
    
    # - 데이터 로드: datasets.py 연동
    datasets = get_clean_datasets()
    bat_df = datasets['bat_tracking']
    
    # - 도메인 피처 통합: feature_engineering.py 연동
    df = calculate_pitcher_stamina_decay(bat_df, baseline_pitches=15)
    df = integrate_catcher_blocking(df, datasets['blocking'])
    df = integrate_fielding_oaa(df, datasets['oaa'])
    
    # - 데이터 정제: JSON 직렬화 및 DB 호환을 위한 NaN -> None(NULL) 변환
    # - 특성: Pandas NaN은 JSON 업로드 시 에러를 유발하므로 float/int/object 구분 없이 처리
    df = df.where(pd.notnull(df), None)
    
    print(f"✅ 마스터 데이터 가공 완료: {df.shape[0]} 행, {df.shape[1]} 개 컬럼")
    return df

def upload_in_batches(
    supabase: Client, 
    df: pd.DataFrame, 
    table_name: str = "statcast_bat_tracking", 
    chunk_size: int = 1000,
    pilot_mode: bool = True
):
    """
    [대용량 벌크 업로드 파이프라인]
    - 목적: 129개 컬럼 대용량 데이터를 메모리 에러와 속도 지연 없이 분할 적재
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
        
        # - 재시도 로직 설정: 네트워크 차단 및 레이트 리밋 대비 최대 5회 재시도
        max_retries = 5
        retry_delay = 2
        success = False
        
        for attempt in range(max_retries):
            try:
                # - Supabase API 호출: bulk insert 실행
                supabase.table(table_name).insert(chunk_data).execute()
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
    
    # - 3. 파일럿 업로드 구동 (검증 목적 1개 청크 테스트)
    # - 실전 업로드 시 pilot_mode=False로 호출 권장
    upload_in_batches(
        supabase=supabase_client,
        df=master_df,
        table_name="statcast_bat_tracking",
        chunk_size=1000,
        pilot_mode=True
    )
