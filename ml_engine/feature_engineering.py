import pandas as pd
import numpy as np
from ml_engine.datasets import get_clean_datasets

def calculate_pitcher_stamina_decay(df: pd.DataFrame, baseline_pitches: int = 15) -> pd.DataFrame:
    """
    [투수 체력 저하 가중치 알고리즘]
    - 목적: AI 모델의 투수 체력 상태 인지
    - 방법: 경기 초반 평균 구속/회전수 기준점 설정 후 실시간 감쇠율 계산
    
    Args:
        df: 타구 추적(bat tracking) 데이터프레임
        baseline_pitches: 기준점 계산용 경기 초반 투구 수 (기본값: 15구)
    """
    print(f"⚾️ 투수 체력 저하 피처 엔지니어링 시작 (기준 투구 수: {baseline_pitches}구)...")
    
    # 원본 데이터 보호 (복사본 생성)
    df_feat = df.copy()
    
    # 1. [데이터 정렬]
    # - 투구 수 누적 계산을 위한 선행 작업
    # - 정렬 기준: 경기(game_pk) -> 투수(pitcher) -> 타석(at_bat_number) -> 투구(pitch_number)
    df_feat = df_feat.sort_values(by=['game_pk', 'pitcher', 'at_bat_number', 'pitch_number']).reset_index(drop=True)
    
    # 2. [경기 내 투구 수 누적 계산]
    # - 대용량 처리를 위해 Pandas groupby, cumcount 벡터화 연산 적용
    # - 경기별/투수별 누적 투구 수 산출 (+1로 1구부터 시작)
    df_feat['pitch_count_in_game'] = df_feat.groupby(['game_pk', 'pitcher']).cumcount() + 1
    
    # 3. [베이스라인 추출]
    # - 기준점: 투수별 해당 경기 첫 N구(baseline_pitches)
    # - 산출 지표: 평균 구속(release_speed), 평균 회전수(release_spin_rate)
    baseline_mask = df_feat['pitch_count_in_game'] <= baseline_pitches
    baseline_df = df_feat[baseline_mask].groupby(['game_pk', 'pitcher'])[['release_speed', 'release_spin_rate']].mean().reset_index()
    
    # - 컬럼명 직관적 변경 (base_speed, base_spin)
    baseline_df = baseline_df.rename(columns={
        'release_speed': 'base_speed',
        'release_spin_rate': 'base_spin'
    })
    
    # 4. [데이터 병합 (Join)]
    # - 산출된 베이스라인 데이터를 원본 데이터프레임에 병합 (Left Join)
    df_feat = df_feat.merge(baseline_df, on=['game_pk', 'pitcher'], how='left')
    
    # 5. [실시간 감쇠 지표 계산 (Decay Ratio)]
    # - 현재 구속/회전수를 베이스라인 대비 비율로 계산 (1.0 = 유지, 0.95 = 5% 하락)
    df_feat['velocity_decay_ratio'] = df_feat['release_speed'] / df_feat['base_speed']
    df_feat['spin_decay_ratio'] = df_feat['release_spin_rate'] / df_feat['base_spin']
    
    # - 예외 처리: 데이터 누락 또는 기준점 미달 시 1.0(정상)으로 결측치 보정
    df_feat['velocity_decay_ratio'] = df_feat['velocity_decay_ratio'].fillna(1.0)
    df_feat['spin_decay_ratio'] = df_feat['spin_decay_ratio'].fillna(1.0)
    
    # 6. [체력 지수(Stamina Index) 계산]
    # - 하락폭 계산 (1.0 - 감쇠율)
    vel_drop = 1.0 - df_feat['velocity_decay_ratio']
    spin_drop = 1.0 - df_feat['spin_decay_ratio']
    
    # - 보정: 구속/회전수가 상승한 경우(음수 발생) 0으로 처리 (체력 저하 없음)
    vel_drop = np.maximum(vel_drop, 0)
    spin_drop = np.maximum(spin_drop, 0)
    
    # - 종합 지수 산출식: (투구 수 / 100) * (구속 하락폭*0.7 + 회전수 하락폭*0.3)
    # - 특성: 투구 수가 많고 구속 하락이 클수록 지수 급증
    df_feat['stamina_index'] = (df_feat['pitch_count_in_game'] / 100.0) * (vel_drop * 0.7 + spin_drop * 0.3)
    
    print("✅ 투수 체력 저하 피처 엔지니어링 완료!")
    return df_feat

if __name__ == "__main__":
    # 검증: 정제 데이터 로드 및 적용
    print("로컬 검증 데이터 로드 중...")
    datasets = get_clean_datasets()
    bat_df = datasets['bat_tracking']
    
    # 함수 실행
    feat_df = calculate_pitcher_stamina_decay(bat_df, baseline_pitches=15)
    
    # 파생 변수 확인 (투구 수 80구 초과 샘플)
    print("\n🔍 [검증] 파생 변수 생성 결과 (상위 10개 행):")
    cols_to_show = [
        'game_pk', 'pitcher', 'pitch_count_in_game', 
        'release_speed', 'base_speed', 'velocity_decay_ratio', 
        'stamina_index'
    ]
    sample_view = feat_df[feat_df['pitch_count_in_game'] > 80][cols_to_show].head(10)
    print(sample_view)
