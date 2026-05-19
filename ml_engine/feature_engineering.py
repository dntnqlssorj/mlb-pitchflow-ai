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

def integrate_catcher_blocking(bat_tracking_df: pd.DataFrame, blocking_df: pd.DataFrame) -> pd.DataFrame:
    """
    [포수 블로킹 데이터 결합 및 위기 상황 파생 변수 생성]
    - 목적: 위기 상황 시 포수 블로킹 능력에 따른 떨어지는 변화구 가중치 산출
    - 방법: fielder_2(포수) 및 game_year 기준으로 블로킹 데이터 조인 후 지표 연산
    """
    print("🧤 포수 블로킹 도메인 로직 결합 시작...")
    # - 원본 데이터 보호 (복사본 생성)
    df_feat = bat_tracking_df.copy()
    
    # - 포수 데이터 결합: fielder_2 및 game_year 기준 레프트 조인
    blocking_sub = blocking_df[['player_id', 'game_year', 'catcher_blocking_runs']].copy()
    blocking_sub = blocking_sub.rename(columns={'player_id': 'fielder_2'})
    df_feat = df_feat.merge(blocking_sub, on=['fielder_2', 'game_year'], how='left')
    
    # - 결측치 처리: 블로킹 지표가 없는 포수는 평균치인 0으로 일괄 대체
    df_feat['catcher_blocking_runs'] = df_feat['catcher_blocking_runs'].fillna(0)
    
    # - 득점권 상황 파생 변수 생성: 2루(on_2b) 또는 3루(on_3b) 주자가 존재하면 득점권(1), 아니면 평시(0)로 식별
    df_feat['is_risp'] = ((df_feat['on_2b'] != 0) | (df_feat['on_3b'] != 0)).astype(int)
    
    # - 블로킹 레버리지 팩터 산출: 득점권 상황에서 포수의 블로킹 런스에 비례해 가중치 부여
    # - 특성: 포수 블로킹 능력이 좋을수록 양수 값 상승 -> 떨어지는 공(포크/커브 등) 구사 확률 증가 힌트
    df_feat['blocking_leverage_factor'] = df_feat['is_risp'] * df_feat['catcher_blocking_runs'] * 0.1
    
    print("✅ 포수 블로킹 결합 완료!")
    return df_feat

def integrate_fielding_oaa(bat_tracking_df: pd.DataFrame, oaa_df: pd.DataFrame) -> pd.DataFrame:
    """
    [야수 OAA 데이터 결합 및 수비 리스크 파생 변수 생성]
    - 목적: 당일 출전 야수진 수비력에 따른 탈삼진 유도 확률 가중치 산출
    - 방법: 출전 중인 야수 7명의 OAA 합산 후 수비 리스크 인덱스 생성
    """
    print("🔄 야수 OAA 도메인 로직 결합 시작...")
    # - 원본 데이터 보호 (복사본 생성)
    df_feat = bat_tracking_df.copy()
    
    # - 야수 OAA 딕셔너리 매핑 맵 생성 (빠른 연산을 위함)
    # - player_id와 game_year를 복합키 인덱스로 설정하여 빠른 벡터화 매핑 준비
    oaa_series = oaa_df.set_index(['player_id', 'game_year'])['outs_above_average']
    
    # - 야수진 ID 컬럼 리스트 (1루수부터 우익수까지 총 7명)
    fielder_cols = ['fielder_3', 'fielder_4', 'fielder_5', 'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9']
    
    # - 당일 출전 야수진 OAA 합산 연산
    df_feat['team_oaa_total'] = 0.0
    for col in fielder_cols:
        # - 각 수비수 위치별로 OAA 점수 매핑 (결측치는 0으로 처리하여 단순 합산)
        idx = pd.MultiIndex.from_arrays([df_feat[col], df_feat['game_year']])
        df_feat['team_oaa_total'] += idx.map(oaa_series).fillna(0)
        
    # - 수비 가변 가중치 생성: 팀 OAA가 낮을수록 탈삼진 리스크 인덱스 증가
    # - 특성: 팀 OAA 총합이 마이너스(수비 불안)일수록 탈삼진을 잡아야 하므로 가중치 상승
    df_feat['fielding_risk_index'] = np.maximum(-df_feat['team_oaa_total'] * 0.05, 0)
    
    print("✅ 야수 OAA 결합 완료!")
    return df_feat

if __name__ == "__main__":
    # 검증: 정제 데이터 로드 및 적용
    print("로컬 검증 데이터 로드 중...")
    datasets = get_clean_datasets()
    bat_df = datasets['bat_tracking']
    blocking_df = datasets['blocking']
    oaa_df = datasets['oaa']
    
    # 1. 투수 체력 저하 피처 생성
    feat_df = calculate_pitcher_stamina_decay(bat_df, baseline_pitches=15)
    
    # 2. 포수 블로킹 결합
    feat_df = integrate_catcher_blocking(feat_df, blocking_df)
    
    # 3. 야수 OAA 결합
    feat_df = integrate_fielding_oaa(feat_df, oaa_df)
    
    # 파생 변수 확인 (투구 수 80구 초과 및 득점권 상황 샘플)
    print("\n🔍 [검증] 전체 파이프라인 결합 완료. 최종 파생 변수 확인 (상위 10개 행):")
    cols_to_show = [
        'game_pk', 'pitcher', 'pitch_count_in_game', 'stamina_index',
        'is_risp', 'blocking_leverage_factor', 
        'team_oaa_total', 'fielding_risk_index'
    ]
    
    # 득점권 상황 및 투구 수가 어느 정도 누적된 행 필터링
    sample_view = feat_df[(feat_df['pitch_count_in_game'] > 60) & (feat_df['is_risp'] == 1)][cols_to_show].head(10)
    if sample_view.empty:
        sample_view = feat_df[cols_to_show].head(10)
        
    print(sample_view)
    print(f"\n최종 DataFrame 형태: {feat_df.shape} (기존 118개 컬럼에서 추가 완료)")
