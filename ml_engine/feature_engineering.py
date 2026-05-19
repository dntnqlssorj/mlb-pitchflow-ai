import pandas as pd
import numpy as np
from ml_engine.datasets import get_clean_datasets

def calculate_pitcher_stamina_decay(df: pd.DataFrame, baseline_pitches: int = 15) -> pd.DataFrame:
    """
    [투수 체력 저하 가중치 알고리즘]
    투수의 경기 초반(1~15구) 평균 구속/회전수를 기준점(Baseline)으로 잡고,
    투구 수가 늘어남에 따라 현재 구속/회전수가 얼마나 떨어졌는지(Decay)를 계산하여
    AI 모델이 투수의 체력 상태를 인지할 수 있도록 파생 변수를 생성합니다.
    
    Args:
        df: 타구 추적(bat tracking) 데이터프레임
        baseline_pitches: 기준점을 계산할 경기 초반 투구 수 (기본값: 15구)
    """
    print(f"⚾️ 투수 체력 저하 피처 엔지니어링 시작 (기준 투구 수: {baseline_pitches}구)...")
    
    # 원본 데이터 보호를 위해 복사본 사용
    df_feat = df.copy()
    
    # 1. [데이터 정렬]
    # 어떤 투수가 어떤 경기에서 던진 공인지 순서대로 나열해야 투구 수를 셀 수 있습니다.
    # 경기 고유번호(game_pk), 투수(pitcher), 타석 번호(at_bat_number), 투구 번호(pitch_number) 순으로 정렬합니다.
    df_feat = df_feat.sort_values(by=['game_pk', 'pitcher', 'at_bat_number', 'pitch_number']).reset_index(drop=True)
    
    # 2. [경기 내 투구 수 누적 계산]
    # for문(루프)을 쓰면 144만 행 처리 시 몇 시간이 걸립니다.
    # 판다스의 groupby와 cumcount를 사용해 경기별/투수별 투구 수를 1초 만에 셉니다. (+1을 해서 1구부터 시작)
    df_feat['pitch_count_in_game'] = df_feat.groupby(['game_pk', 'pitcher']).cumcount() + 1
    
    # 3. [베이스라인 추출]
    # 각 투수가 해당 경기에서 처음 던진 N구(baseline_pitches)의 평균 구속과 회전수를 구합니다.
    # 이게 바로 오늘 이 투수의 '쌩쌩할 때 컨디션' 기준점입니다.
    baseline_mask = df_feat['pitch_count_in_game'] <= baseline_pitches
    baseline_df = df_feat[baseline_mask].groupby(['game_pk', 'pitcher'])[['release_speed', 'release_spin_rate']].mean().reset_index()
    
    # 병합 후 헷갈리지 않게 컬럼명을 'base_speed', 'base_spin'으로 바꿔줍니다.
    baseline_df = baseline_df.rename(columns={
        'release_speed': 'base_speed',
        'release_spin_rate': 'base_spin'
    })
    
    # 4. [데이터 병합 (Join)]
    # 계산된 기준점(baseline) 데이터를 원본 전체 데이터에 경기별/투수별로 맵핑(Left Join)합니다.
    df_feat = df_feat.merge(baseline_df, on=['game_pk', 'pitcher'], how='left')
    
    # 5. [실시간 감쇠 지표 계산 (Decay Ratio)]
    # 현재 구속을 쌩쌩할 때 구속으로 나눕니다. 
    # 1.0이면 그대로, 0.95면 구속이 5% 떨어졌다는 뜻입니다. (벡터화 연산으로 순식간에 계산됨)
    df_feat['velocity_decay_ratio'] = df_feat['release_speed'] / df_feat['base_speed']
    df_feat['spin_decay_ratio'] = df_feat['release_spin_rate'] / df_feat['base_spin']
    
    # 결측치 방어 코드: 투수가 공을 너무 적게 던졌거나 데이터가 누락되어 기준점이 없으면 1.0(정상)으로 채워줍니다.
    df_feat['velocity_decay_ratio'] = df_feat['velocity_decay_ratio'].fillna(1.0)
    df_feat['spin_decay_ratio'] = df_feat['spin_decay_ratio'].fillna(1.0)
    
    # 6. [체력 지수(Stamina Index) 정의]
    # AI 모델이 투수의 체력이 얼마나 빠졌는지 직관적으로 알 수 있는 종합 점수입니다.
    # 공식: (투구 수 / 100) * (구속 저하 폭 * 가중치 + 회전수 저하 폭 * 가중치)
    # 구속이 많이 떨어질수록, 투구 수가 100구에 가까워질수록 이 지수는 급격히 커집니다.
    vel_drop = 1.0 - df_feat['velocity_decay_ratio']
    spin_drop = 1.0 - df_feat['spin_decay_ratio']
    
    # np.maximum을 써서 구속이 오히려 올랐을 경우(음수)를 0으로 보정해 줍니다.
    vel_drop = np.maximum(vel_drop, 0)
    spin_drop = np.maximum(spin_drop, 0)
    
    # 체력 지수: 투구 수가 많고 구속/회전이 떨어지면 값이 커짐 (스태미나 소진 상태)
    df_feat['stamina_index'] = (df_feat['pitch_count_in_game'] / 100.0) * (vel_drop * 0.7 + spin_drop * 0.3)
    
    print("✅ 투수 체력 저하 피처 엔지니어링 완료!")
    return df_feat

if __name__ == "__main__":
    # 1. 원본 데이터 로드 (앞서 만든 정제 모듈 활용)
    # 테스트를 위해 전체 데이터 대신 bat_tracking만 가져오거나 일부만 로드합니다.
    # get_clean_datasets는 시간이 걸리므로 bat_tracking만 로드하는 모듈을 직접 부를 수도 있지만,
    # 요구사항에 맞게 전체를 불러와 그 중 bat_tracking을 사용합니다.
    print("로컬 검증을 위해 데이터를 로드합니다...")
    datasets = get_clean_datasets()
    bat_df = datasets['bat_tracking']
    
    # 2. 피처 엔지니어링 함수 실행
    feat_df = calculate_pitcher_stamina_decay(bat_df, baseline_pitches=15)
    
    # 3. 결과 검증 (새로 생성된 파생 변수들만 뽑아서 출력)
    print("\n🔍 [검증] 파생 변수 생성 결과 (상위 10개 행):")
    cols_to_show = [
        'game_pk', 'pitcher', 'pitch_count_in_game', 
        'release_speed', 'base_speed', 'velocity_decay_ratio', 
        'stamina_index'
    ]
    # 변화를 보기 위해 투구 수가 어느 정도 쌓인 후반부 투구 데이터를 샘플로 봅니다.
    sample_view = feat_df[feat_df['pitch_count_in_game'] > 80][cols_to_show].head(10)
    print(sample_view)
