import pandas as pd
import os
from pathlib import Path

# MLB 데이터 최상위 경로를 지정합니다. (현재 프로젝트 폴더 기준)
# Path 객체를 사용하면 맥, 윈도우 상관없이 슬래시(/)로 안전하게 경로를 이을 수 있습니다.
BASE_DIR = Path("data/MLB All Data")

def load_framing_data() -> pd.DataFrame:
    """
    [포수 프레이밍 데이터 병합 및 정제 함수]
    포수가 스트라이크 존 바깥쪽 공을 스트라이크로 만드는 능력(프레이밍)을 담은 데이터를 불러옵니다.
    """
    # 1. 2024년, 2025년 CSV 파일을 각각 읽어옵니다.
    df_2024 = pd.read_csv(BASE_DIR / "Catcher" / "catcher-framing.csv")
    # 나중에 두 데이터를 합쳤을 때 어느 연도 데이터인지 구분하기 위해 'game_year'라는 열(컬럼)을 강제로 추가합니다.
    df_2024['game_year'] = 2024
    
    df_2025 = pd.read_csv(BASE_DIR / "Catcher" / "catcher-framing-2.csv")
    df_2025['game_year'] = 2025
    
    # 2. pd.concat을 사용해 두 데이터프레임을 위아래로 이어 붙입니다.
    # ignore_index=True를 하면 합쳐진 데이터의 인덱스 번호가 0부터 다시 깔끔하게 매겨집니다.
    df = pd.concat([df_2024, df_2025], ignore_index=True)
    
    # 3. [데이터 클렌징] 컬럼명 통일
    # 프레이밍 데이터만 선수 ID 컬럼명이 'id'로 되어 있습니다.
    # 나중에 다른 데이터(블로킹, OAA 등)와 엮기(JOIN) 위해 통일성 있게 'player_id'로 바꿔줍니다.
    if 'id' in df.columns:
        df = df.rename(columns={'id': 'player_id'})
        
    return df

def load_blocking_data() -> pd.DataFrame:
    """
    [포수 블로킹 데이터 병합 및 정제 함수]
    투수가 던진 폭투나 바운드볼을 포수가 얼마나 잘 막아냈는지(블로킹)에 대한 데이터를 불러옵니다.
    """
    # 2024년 데이터 로드 및 game_year 컬럼 생성
    df_2024 = pd.read_csv(BASE_DIR / "Catcher" / "catcher_blocking.csv")
    df_2024['game_year'] = 2024
    
    # 2025년 데이터 로드 및 game_year 컬럼 생성
    df_2025 = pd.read_csv(BASE_DIR / "Catcher" / "catcher_blocking-2.csv")
    df_2025['game_year'] = 2025
    
    # 두 데이터를 하나로 합칩니다.
    df = pd.concat([df_2024, df_2025], ignore_index=True)
    
    # 블로킹 데이터는 별도의 컬럼명 수정이나 결측치 처리가 필요 없을 정도로 깔끔해서 바로 반환합니다.
    return df

def load_oaa_data() -> pd.DataFrame:
    """
    [야수 OAA(Outs Above Average) 데이터 병합 및 정제 함수]
    야수들의 수비 범위를 측정하여 평균 대비 얼마나 많은 아웃카운트를 잡아냈는지 보여주는 데이터입니다.
    """
    # 2024, 2025 데이터 로드 및 연도 표시 컬럼 생성
    df_2024 = pd.read_csv(BASE_DIR / "Depense" / "outs_above_average.csv")
    df_2024['game_year'] = 2024
    
    df_2025 = pd.read_csv(BASE_DIR / "Depense" / "outs_above_average-2.csv")
    df_2025['game_year'] = 2025
    
    # 하나의 표로 합칩니다.
    df = pd.concat([df_2024, df_2025], ignore_index=True)
    
    # 3. [데이터 클렌징] 퍼센트(%) 문자열을 숫자로 변환
    # 문제점: OAA 데이터의 일부 성공률 지표가 숫자(73.5)가 아니라 문자열("73.5%") 형태로 들어있습니다.
    # 컴퓨터는 "73.5%"를 글자로 인식하므로 덧셈 뺄셈 같은 머신러닝 연산을 할 수 없습니다.
    pct_cols = [
        'actual_success_rate_formatted', 
        'adj_estimated_success_rate_formatted', 
        'diff_success_rate_formatted'
    ]
    for col in pct_cols:
        if col in df.columns:
            # 1단계: 글자 속에서 '%' 기호를 찾아 없애고(replace), 혹시 모를 양옆 공백을 지워줍니다(strip).
            df[col] = df[col].astype(str).str.replace('%', '').str.strip()
            # 2단계: 깔끔해진 글자를 진짜 숫자(float)로 바꿔줍니다. 
            # errors='coerce'는 숫자로 바꿀 수 없는 이상한 값이 있으면 강제로 빈칸(NaN)으로 만든다는 뜻입니다.
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def load_bat_tracking_data() -> pd.DataFrame:
    """
    [타구 추적(Bat Tracking) 데이터 로드 및 정제 함수]
    투구별 속도, 타구 속도, 스윙 궤적 등 가장 방대한 데이터가 들어있는 메인 파일입니다.
    """
    # 타구 추적 데이터는 이미 2024년과 2025년이 하나의 파일로 합쳐져 다운로드 되었습니다.
    # 파일 용량이 매우 크지만 판다스는 똑같이 읽어올 수 있습니다.
    df = pd.read_csv(BASE_DIR / "All 24 25 Player" / "statcast_bat_tracking_2024_2025.csv")
    
    # 3. [데이터 클렌징] 주자 상황 결측치(Null) 처리
    # 타구 데이터에는 1루(on_1b), 2루(on_2b), 3루(on_3b)에 주자가 있는지 적혀있습니다.
    # 만약 베이스에 주자가 없다면 빈칸(Null/NaN)으로 저장되어 있습니다.
    # 머신러닝 모델은 빈칸을 만나면 에러를 뿜기 때문에, 주자가 없다는 뜻으로 '0'을 채워 넣어줍니다.
    runner_cols = ['on_1b', 'on_2b', 'on_3b']
    for col in runner_cols:
        if col in df.columns:
            # fillna(0): 빈칸을 모두 0으로 덮어씌웁니다.
            df[col] = df[col].fillna(0)
            
    return df

def get_clean_datasets() -> dict:
    """
    위에서 만든 4개의 정제 함수를 한 번에 실행하여 
    딕셔너리(사전) 형태로 묶어 반환해 주는 마스터 함수입니다.
    """
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
    
    # 키(이름) : 값(정제된 데이터프레임) 형태로 묶어 반환합니다.
    return {
        'framing': framing_df,
        'blocking': blocking_df,
        'oaa': oaa_df,
        'bat_tracking': bat_tracking_df
    }

if __name__ == "__main__":
    # 이 파일이 자체적으로 실행될 때(검증용) 작동하는 구역입니다.
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
