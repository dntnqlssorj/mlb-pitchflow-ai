import pandas as pd
import os
from pathlib import Path

# MLB 데이터 최상위 경로 (루트 디렉토리 기준)
BASE_DIR = Path("data/MLB All Data")

def load_framing_data() -> pd.DataFrame:
    """포수 프레이밍 데이터 병합 및 정제"""
    df_2024 = pd.read_csv(BASE_DIR / "Catcher" / "catcher-framing.csv")
    df_2024['game_year'] = 2024
    
    df_2025 = pd.read_csv(BASE_DIR / "Catcher" / "catcher-framing-2.csv")
    df_2025['game_year'] = 2025
    
    # 두 년도 데이터 병합
    df = pd.concat([df_2024, df_2025], ignore_index=True)
    
    # id를 player_id로 변경 (정합성 통일)
    if 'id' in df.columns:
        df = df.rename(columns={'id': 'player_id'})
        
    return df

def load_blocking_data() -> pd.DataFrame:
    """포수 블로킹 데이터 병합 및 정제"""
    df_2024 = pd.read_csv(BASE_DIR / "Catcher" / "catcher_blocking.csv")
    df_2024['game_year'] = 2024
    
    df_2025 = pd.read_csv(BASE_DIR / "Catcher" / "catcher_blocking-2.csv")
    df_2025['game_year'] = 2025
    
    df = pd.concat([df_2024, df_2025], ignore_index=True)
    
    return df

def load_oaa_data() -> pd.DataFrame:
    """야수 OAA 데이터 병합 및 정제"""
    df_2024 = pd.read_csv(BASE_DIR / "Depense" / "outs_above_average.csv")
    df_2024['game_year'] = 2024
    
    df_2025 = pd.read_csv(BASE_DIR / "Depense" / "outs_above_average-2.csv")
    df_2025['game_year'] = 2025
    
    df = pd.concat([df_2024, df_2025], ignore_index=True)
    
    # % 기호 제거 후 float 형변환
    pct_cols = [
        'actual_success_rate_formatted', 
        'adj_estimated_success_rate_formatted', 
        'diff_success_rate_formatted'
    ]
    for col in pct_cols:
        if col in df.columns:
            # 안전한 캐스팅 (결측치는 NaN으로 자동 처리)
            df[col] = df[col].astype(str).str.replace('%', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def load_bat_tracking_data() -> pd.DataFrame:
    """타구 추적 데이터 로드 및 정제"""
    # 원본 파일이 24_25 병합 형태이며 용량이 크므로 필요한 전처리 수행
    # 로컬 메모리를 고려하여 engine='c' 사용
    df = pd.read_csv(BASE_DIR / "All 24 25 Player" / "statcast_bat_tracking_2024_2025.csv")
    
    # 주자 상황(on_1b, on_2b, on_3b) Null 값을 0으로 치환 (빈 루상 표기)
    runner_cols = ['on_1b', 'on_2b', 'on_3b']
    for col in runner_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    return df

def get_clean_datasets() -> dict:
    """모든 정제된 데이터셋을 딕셔너리로 반환"""
    print("🔄 데이터 병합 및 정제 시작...")
    
    print(" - 프레이밍(Framing) 데이터 처리 중...")
    framing_df = load_framing_data()
    
    print(" - 블로킹(Blocking) 데이터 처리 중...")
    blocking_df = load_blocking_data()
    
    print(" - 야수 OAA 데이터 처리 중...")
    oaa_df = load_oaa_data()
    
    print(" - 타구 추적(Bat Tracking) 데이터 처리 중... (대용량, 시간이 걸릴 수 있습니다)")
    bat_tracking_df = load_bat_tracking_data()
    
    print("✅ 모든 데이터 전처리 완료!")
    return {
        'framing': framing_df,
        'blocking': blocking_df,
        'oaa': oaa_df,
        'bat_tracking': bat_tracking_df
    }

if __name__ == "__main__":
    # 로컬 검증
    datasets = get_clean_datasets()
    print("\n📊 데이터셋 검증 결과:")
    for name, df in datasets.items():
        print(f"[{name.upper()}] Row/Col: {df.shape}")
        
        if name == 'framing':
            print(f"  > 'player_id' 컬럼 존재 여부: {'player_id' in df.columns} (기존 id 컬럼 변경 확인)")
        
        elif name == 'oaa':
            sample_val = df['actual_success_rate_formatted'].iloc[0]
            print(f"  > OAA 성공률 지표 타입: {type(sample_val)} (Float 캐스팅 확인)")
        
        elif name == 'bat_tracking':
            null_count = df[['on_1b', 'on_2b', 'on_3b']].isnull().sum().sum()
            print(f"  > 주자 결측치 총합: {null_count} (Null->0 치환 확인)")
            
        print(f"  > 'game_year' 컬럼 정상 반영 여부: {'game_year' in df.columns}")
        print("-" * 40)
