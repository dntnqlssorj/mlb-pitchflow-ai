# ==============================================================================
# MLB PitchFlow AI - 도메인 피처 엔지니어링
# 변경 이력: 2026-05-21 Target Leakage 제거 (research.md §2.4 기준)
#   - build_season_baseline() 신설: 시즌 집계 베이스라인 사전 산출
#   - calculate_pitcher_stamina_decay() 재설계:
#       경기 내 첫 N구 평균 → 시즌 집계 + shift(1) rolling 이동 평균으로 전환
#       현재 투구의 release_speed/release_spin_rate 직접 참조 완전 배제
# ==============================================================================

import pandas as pd
import numpy as np
from ml_engine.config import STAMINA_BASELINE_PITCHES


def build_season_baseline(bat_df: pd.DataFrame) -> pd.DataFrame:
    """
    [시즌 집계 베이스라인 산출]
    - 목적: 투수별 시즌 전체 평균 구속/회전수를 사전 집계하여 stamina 계산의
            누수 없는 기준점(reference) 제공
    - 호출 위치: train.py의 샘플링(sample()) 이전 단계에서 전체 bat_df 대상으로 반드시 호출
    - 누수 근거: 경기 내 첫 N구 평균은 현재 투구의 release_speed를 포함하므로 100% 누수.
                시즌 집계값은 과거 전체 등판 데이터 기반이므로 투구 이전 시점에 확정된 값.

    Args:
        bat_df: 전체(미샘플링) bat_tracking DataFrame
                필수 컬럼: pitcher, game_year, release_speed, release_spin_rate

    Returns:
        DataFrame: [pitcher, game_year, season_avg_speed, season_avg_spin]
    """
    print("시즌 집계 베이스라인 산출 중 (전체 데이터 기준)...")

    baseline = (
        bat_df
        .groupby(['pitcher', 'game_year'])
        .agg({
            'release_speed': 'mean',
            'release_spin_rate': 'mean',
            'p_throws': 'first',
        })
        .reset_index()
        .rename(columns={
            'release_speed':     'season_avg_speed',
            'release_spin_rate': 'season_avg_spin',
        })
    )

    print(f"베이스라인 산출 완료: {len(baseline)}개 (투수 × 시즌) 페어")
    return baseline


def calculate_pitcher_stamina_decay(
    df: pd.DataFrame,
    season_baseline_df: pd.DataFrame,
    baseline_pitches: int = STAMINA_BASELINE_PITCHES,
) -> pd.DataFrame:
    """
    [투수 체력 저하 피처 엔지니어링 — 누수 제거 버전]
    - 목적: AI 모델의 투수 체력 상태 인지
    - 변경 전: base_speed = 동일 경기 내 첫 N구 release_speed 평균 (현재 투구 포함 → 누수)
    - 변경 후: base_speed = 시즌 전체 집계 평균 (season_baseline_df에서 조인, 누수 없음)
               velocity_decay_ratio = shift(1) rolling 평균 / base_speed
               → shift(1): 현재 투구 자신을 이동 평균에서 배제하는 핵심 누수 차단 장치

    Args:
        df: (샘플링된) bat_tracking DataFrame
        season_baseline_df: build_season_baseline() 반환값
                            컬럼: [pitcher, game_year, season_avg_speed, season_avg_spin]
        baseline_pitches: rolling 이동 평균 window 크기 (기본값 15, 파라미터명 유지)

    Returns:
        DataFrame: 원본 + [pitch_count_in_game, base_speed, base_spin,
                           velocity_decay_ratio, spin_decay_ratio, stamina_index]
    """
    print(f"투수 체력 저하 피처 엔지니어링 시작 (rolling window: {baseline_pitches}구)...")

    df_feat = df.copy()

    # ------------------------------------------------------------------
    # 단계 1. 시간 순서 정렬
    # cumcount 및 rolling 연산의 시간 순서 보장
    # ------------------------------------------------------------------
    df_feat = df_feat.sort_values(
        by=['game_pk', 'pitcher', 'at_bat_number', 'pitch_number']
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 단계 2. 경기 내 누적 투구 수 산출 (누수 없음)
    # pitch_count_in_game: 현재 투구 이전까지의 누적 수 + 1
    # groupby 후 cumcount는 0-indexed → +1로 1구부터 시작
    # ------------------------------------------------------------------
    df_feat['pitch_count_in_game'] = (
        df_feat.groupby(['game_pk', 'pitcher']).cumcount() + 1
    )

    # ------------------------------------------------------------------
    # 단계 3. 시즌 베이스라인 병합 (경기 내 집계 방식 완전 대체)
    # season_baseline_df를 (pitcher, game_year) 기준 LEFT JOIN
    # base_speed, base_spin: 시즌 전체 평균 → 투구 이전 시점 확정값
    # ------------------------------------------------------------------
    df_feat = df_feat.merge(
        season_baseline_df.rename(columns={
            'season_avg_speed': 'base_speed',
            'season_avg_spin':  'base_spin',
        }),
        on=['pitcher', 'game_year'],
        how='left',
    )

    # ------------------------------------------------------------------
    # 단계 4. Rolling 이동 평균 구속·회전수 산출 (핵심 누수 차단 장치)
    # shift(1): 현재 투구 자신(row N)을 이동 평균 계산에서 배제
    #           → row N의 rolling 평균 = rows (N-window)~(N-1)의 평균
    # min_periods=1: 초반 투구(1~baseline_pitches-1구)에서도 가용 데이터로 계산
    # ------------------------------------------------------------------
    grp = df_feat.groupby(['game_pk', 'pitcher'])

    rolling_speed = grp['release_speed'].transform(
        lambda s: s.shift(1).rolling(window=baseline_pitches, min_periods=1).mean()
    )
    rolling_spin = grp['release_spin_rate'].transform(
        lambda s: s.shift(1).rolling(window=baseline_pitches, min_periods=1).mean()
    )

    # ------------------------------------------------------------------
    # 단계 5. 감쇠율 재정의 (누수 차단)
    # 분자: 직전 투구까지의 rolling 이동 평균 (현재 투구 미포함)
    # 분모: 시즌 전체 평균 구속/회전수 (정적 집계값)
    # 결측 보정: 분모 0 또는 NaN → 1.0 대체 (정상 상태로 처리)
    # ------------------------------------------------------------------
    df_feat['velocity_decay_ratio'] = (rolling_speed / df_feat['base_speed']).fillna(1.0)
    df_feat['spin_decay_ratio']     = (rolling_spin  / df_feat['base_spin']).fillna(1.0)

    # base_speed, base_spin 자체 결측 처리 (신인 투수 등 데이터 미적재 케이스)
    df_feat['base_speed'] = df_feat['base_speed'].fillna(0.0)
    df_feat['base_spin']  = df_feat['base_spin'].fillna(0.0)

    # ------------------------------------------------------------------
    # 단계 6. stamina_index 재산출
    # 산출식 구조 유지: (투구 수 / 100) * (구속 하락폭*0.7 + 회전수 하락폭*0.3)
    # vel_drop, spin_drop: 음수 방지 클리핑 (구속/회전수가 오히려 상승한 경우 0 처리)
    # ------------------------------------------------------------------
    vel_drop  = np.maximum(1.0 - df_feat['velocity_decay_ratio'], 0)
    spin_drop = np.maximum(1.0 - df_feat['spin_decay_ratio'],     0)

    df_feat['stamina_index'] = (
        (df_feat['pitch_count_in_game'] / 100.0) * (vel_drop * 0.7 + spin_drop * 0.3)
    )

    print("투수 체력 저하 피처 엔지니어링 완료")
    return df_feat


def add_situational_features(df):
    """
    [상황별 파생 변수 추가]
    - count_situation: 3*balls + strikes (볼카운트 상황을 단일 수치로 인코딩)
    - matchup_type: 타자(stand)와 투수(p_throws)의 좌우 매치업 (LL=0, LR=1, RL=2, RR=3, 그외 4)
    """
    df['count_situation'] = df['balls'] * 3 + df['strikes']
    if 'p_throws' in df.columns:
        matchup_map = {'LL':0,'LR':1,'RL':2,'RR':3}
        df['matchup_type'] = (
            df['stand'].astype(str) + df['p_throws'].astype(str)
        ).map(matchup_map).fillna(4).astype(int)
    else:
        df['matchup_type'] = 4
    return df


def integrate_catcher_blocking(
    bat_tracking_df: pd.DataFrame,
    blocking_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    [포수 블로킹 데이터 결합 및 위기 상황 파생 변수 생성 — 변경 없음]
    - 변경 사항: 없음. is_risp, blocking_leverage_factor 모두 투구 이전 주자 상황 기반이므로 누수 없음.

    Args:
        bat_tracking_df: 타구 추적 DataFrame (stamina 피처 추가 후)
        blocking_df:     포수 블로킹 마스터 DataFrame

    Returns:
        DataFrame: 원본 + [catcher_blocking_runs, is_risp, blocking_leverage_factor]
    """
    print("포수 블로킹 도메인 로직 결합 시작...")

    df_feat = bat_tracking_df.copy()

    # 포수 데이터 결합: fielder_2(포수) 및 game_year 기준 LEFT JOIN
    blocking_sub = (
        blocking_df[['player_id', 'game_year', 'catcher_blocking_runs']]
        .copy()
        .rename(columns={'player_id': 'fielder_2'})
    )
    df_feat = df_feat.merge(blocking_sub, on=['fielder_2', 'game_year'], how='left')

    # 결측치 처리: 블로킹 지표 없는 포수 → 리그 평균(0)으로 대체
    df_feat['catcher_blocking_runs'] = df_feat['catcher_blocking_runs'].fillna(0)

    # 득점권 상황 파생 변수: 2루(on_2b) 또는 3루(on_3b) 주자 존재 시 1
    df_feat['is_risp'] = (
        (df_feat['on_2b'] != 0) | (df_feat['on_3b'] != 0)
    ).astype(int)

    # 블로킹 레버리지 팩터: RISP 상황에서 포수 블로킹 능력 가중치
    df_feat['blocking_leverage_factor'] = (
        df_feat['is_risp'] * df_feat['catcher_blocking_runs'] * 0.1
    )

    print("포수 블로킹 결합 완료")
    return df_feat


def integrate_fielding_oaa(
    bat_tracking_df: pd.DataFrame,
    oaa_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    [야수 OAA 데이터 결합 및 수비 리스크 파생 변수 생성 — 변경 없음]
    - 변경 사항: 없음. team_oaa_total, fielding_risk_index 모두 시즌 집계 지표이므로 누수 없음.

    Args:
        bat_tracking_df: 포수 블로킹 피처 추가 후 DataFrame
        oaa_df:          야수 OAA 마스터 DataFrame

    Returns:
        DataFrame: 원본 + [team_oaa_total, fielding_risk_index]
    """
    print("야수 OAA 도메인 로직 결합 시작...")

    df_feat = bat_tracking_df.copy()

    # 야수 OAA 딕셔너리 매핑 (player_id, game_year) 복합키 인덱스
    oaa_series = oaa_df.set_index(['player_id', 'game_year'])['outs_above_average']

    # 야수진 ID 컬럼 (1루수~우익수, 총 7명)
    fielder_cols = ['fielder_3', 'fielder_4', 'fielder_5',
                    'fielder_6', 'fielder_7', 'fielder_8', 'fielder_9']

    # 출전 야수진 OAA 합산
    df_feat['team_oaa_total'] = 0.0
    for col in fielder_cols:
        idx = pd.MultiIndex.from_arrays([df_feat[col], df_feat['game_year']])
        df_feat['team_oaa_total'] += idx.map(oaa_series).fillna(0)

    # 수비 리스크 인덱스: 팀 OAA 마이너스일수록 탈삼진 유도 필요성 증가
    df_feat['fielding_risk_index'] = np.maximum(-df_feat['team_oaa_total'] * 0.05, 0)

    print("야수 OAA 결합 완료")
    return df_feat


def add_batter_swing_tendency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [타자 구종별 스윙 경향 피처 생성]
    - 목적: 투수 구종 선택에 영향을 미치는 타자의 약점/강점 구종 정보를 명시적으로 제공
    - 계산 기준: 동일 타자의 동일 game_year 전체 타석 대상 구종별 스윙율 산출
    - 스윙 판정: description 컬럼이 스윙 액션을 포함하면 1 (swinging_strike, foul, hit_into_play 등)
    - 누수 여부: 시즌 집계 통계 (사전 확정값) → 누수 없음
    - 대상 구종: FF, SL, CH, SI, CU, FC (레퍼토리 피처와 동일 6종)
    """
    print("Batter Swing Tendency 피처 엔지니어링 시작...")

    # --- 스윙 여부 파생 ---
    # statcast description 기준 스윙 액션 식별자
    SWING_DESCRIPTIONS = {
        'swinging_strike', 'swinging_strike_blocked',
        'foul', 'foul_tip', 'foul_bunt',
        'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score',
        'missed_bunt',
    }

    if 'description' in df.columns:
        df['_is_swing'] = df['description'].isin(SWING_DESCRIPTIONS).astype(int)
    else:
        # description 컬럼 없는 경우 (추론 시점): 스윙율 0.5로 대체
        print("  [경고] description 컬럼 없음 — 스윙율 0.5로 전체 대체")
        TARGET_PITCHES = ['FF', 'SL', 'CH', 'SI', 'CU', 'FC']
        RENAME_MAP = {pt: f'batter_{pt.lower()}_swing_rate' for pt in TARGET_PITCHES}
        for col in RENAME_MAP.values():
            df[col] = 0.5
        return df

    TARGET_PITCHES = ['FF', 'SL', 'CH', 'SI', 'CU', 'FC']
    RENAME_MAP = {pt: f'batter_{pt.lower()}_swing_rate' for pt in TARGET_PITCHES}

    for pt in TARGET_PITCHES:
        col_name = RENAME_MAP[pt]
        pt_mask = df['pitch_type'] == pt

        if pt_mask.sum() == 0:
            df[col_name] = 0.5
            continue

        # 타자 × 시즌 × 해당 구종 스윙율
        swing_rate = (
            df[pt_mask]
            .groupby(['batter', 'game_year'])['_is_swing']
            .mean()
            .reset_index()
            .rename(columns={'_is_swing': col_name})
        )

        df = df.merge(swing_rate, on=['batter', 'game_year'], how='left')

        # 해당 구종을 상대한 기록 없는 타자 → 리그 평균 대체
        league_avg = df[col_name].mean()
        df[col_name] = df[col_name].fillna(league_avg if not np.isnan(league_avg) else 0.5)

    df = df.drop(columns=['_is_swing'])

    for pt in TARGET_PITCHES:
        col_name = RENAME_MAP[pt]
        print(f"  {col_name} 평균: {df[col_name].mean():.4f}")

    print("Batter Swing Tendency 피처 엔지니어링 완료")
    return df


if __name__ == "__main__":
    from ml_engine.datasets import get_clean_datasets

    print("로컬 검증: 전체 피처 엔지니어링 파이프라인 실행 중...")
    datasets  = get_clean_datasets()
    bat_df    = datasets['bat_tracking']
    blocking_df = datasets['blocking']
    oaa_df    = datasets['oaa']

    # 시즌 베이스라인 사전 산출 (샘플링 전 전체 데이터 기준)
    season_baseline = build_season_baseline(bat_df)

    # 파이프라인 체인
    feat_df = calculate_pitcher_stamina_decay(bat_df, season_baseline, baseline_pitches=15)
    feat_df = integrate_catcher_blocking(feat_df, blocking_df)
    feat_df = integrate_fielding_oaa(feat_df, oaa_df)

    # 검증 출력: 득점권 상황 + 누적 투구 수 60구 초과 샘플
    cols_to_show = [
        'game_pk', 'pitcher', 'pitch_count_in_game',
        'base_speed', 'velocity_decay_ratio', 'stamina_index',
        'is_risp', 'blocking_leverage_factor',
        'team_oaa_total', 'fielding_risk_index',
    ]
    sample = feat_df[
        (feat_df['pitch_count_in_game'] > 60) & (feat_df['is_risp'] == 1)
    ][cols_to_show].head(10)

    if sample.empty:
        sample = feat_df[cols_to_show].head(10)

    print("\n[검증] 최종 파생 변수 확인 (상위 10개 행):")
    print(sample)
    print(f"\nDataFrame 형태: {feat_df.shape}")
    print(f"base_speed 결측 비율: {feat_df['base_speed'].isna().mean():.4f}")
    print(f"velocity_decay_ratio 결측 비율: {feat_df['velocity_decay_ratio'].isna().mean():.4f}")


def add_pitch_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Pitch Sequence 정형 피처 생성]
    - 목적: 직전 1~3구 구종을 정수 인코딩하여 시퀀스 정보 주입
    - 누수 차단: shift(N) 적용으로 현재 투구 자신 배제
    - 결측 처리: 경기 첫 구 등 이전 투구 없는 케이스 → -1 패딩
    """
    print("Pitch Sequence 피처 엔지니어링 시작...")

    # - 시간 순 정렬: 시퀀스 연산의 순서 보장
    df = df.sort_values(
        ['game_pk', 'pitcher', 'at_bat_number', 'pitch_number']
    ).copy()

    # - 구종 Label Encoding: 문자열 구종 → 정수 (OT 포함 전체 클래스)
    from sklearn.preprocessing import LabelEncoder
    seq_le = LabelEncoder()
    df['pitch_type_encoded'] = seq_le.fit_transform(
        df['pitch_type'].astype(str)
    )

    # - Shift 연산: 현재 투구 배제 후 직전 N구 구종 추출
    grp = df.groupby(['game_pk', 'pitcher'])['pitch_type_encoded']
    df['prev_pitch_1'] = grp.shift(1).fillna(-1).astype(int)
    df['prev_pitch_2'] = grp.shift(2).fillna(-1).astype(int)
    df['prev_pitch_3'] = grp.shift(3).fillna(-1).astype(int)

    print(f"  prev_pitch_1 결측(-1) 비율: "
          f"{(df['prev_pitch_1'] == -1).mean():.2%}")
    print("Pitch Sequence 피처 엔지니어링 완료")

    return df


def add_pitcher_repertoire_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [투수 구종 레퍼토리 비율 피처 생성]
    - 목적: pitcher ID 암기 구조 대체 — 시즌 구종 비율을 명시적 피처로 제공
    - 계산 기준: 동일 투수의 동일 game_year 전체 투구 대상 비율 산출
    - 누수 여부: 시즌 집계 통계 (사전 확정값) → 누수 없음
    """
    print("Pitcher Repertoire 피처 엔지니어링 시작...")

    # - 투수 × 시즌 × 구종 비율 산출
    repertoire = (
        df.groupby(['pitcher', 'game_year'])['pitch_type']
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reset_index()
    )

    TARGET_PITCHES = ['FF', 'SL', 'CH', 'SI', 'CU', 'FC']
    for col in TARGET_PITCHES:
        if col not in repertoire.columns:
            repertoire[col] = 0.0

    RENAME_MAP = {
        'FF': 'pitcher_ff_pct',
        'SL': 'pitcher_sl_pct',
        'CH': 'pitcher_ch_pct',
        'SI': 'pitcher_si_pct',
        'CU': 'pitcher_cu_pct',
        'FC': 'pitcher_fc_pct',
    }
    repertoire = repertoire[['pitcher', 'game_year'] + TARGET_PITCHES].rename(
        columns=RENAME_MAP
    )

    # - 원본 df에 LEFT JOIN
    df = df.merge(repertoire, on=['pitcher', 'game_year'], how='left')

    PCT_COLS = list(RENAME_MAP.values())
    for col in PCT_COLS:
        league_avg = df[col].mean() if col in df.columns else 0.0
        df[col] = df[col].fillna(league_avg)

    print(f"  pitcher_ff_pct 평균: {df['pitcher_ff_pct'].mean():.4f}")
    print(f"  pitcher_sl_pct 평균: {df['pitcher_sl_pct'].mean():.4f}")
    print(f"  pitcher_ch_pct 평균: {df['pitcher_ch_pct'].mean():.4f}")
    print(f"  pitcher_si_pct 평균: {df['pitcher_si_pct'].mean():.4f}")
    print(f"  pitcher_cu_pct 평균: {df['pitcher_cu_pct'].mean():.4f}")
    print(f"  pitcher_fc_pct 평균: {df['pitcher_fc_pct'].mean():.4f}")
    print("Pitcher Repertoire 피처 엔지니어링 완료")

    return df


def add_pitcher_situation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    [투수별 카운트/매치업 상황별 구종 비율 피처 생성]
    - 목적: 시즌 전체 비율(pitcher_ff_pct)의 한계 보완
    - 투수는 카운트/매치업 상황에 따라 구종 선택이 완전히 다름
    - 누수 여부: 시즌 집계 통계 (사전 확정값) → 누수 없음
    """
    print("Pitcher Situation 피처 엔지니어링 시작...")

    # 카운트 상황 분류
    # ahead: 투수 유리 (0-1, 0-2, 1-2)
    # behind: 타자 유리 (1-0, 2-0, 3-0, 2-1, 3-1)
    # even: 동등 (0-0, 1-1, 2-2, 3-2)
    def get_count_situation(row):
        b, s = row['balls'], row['strikes']
        if (b == 0 and s >= 1) or (b == 1 and s == 2):
            return 'ahead'
        elif b >= s + 1:
            return 'behind'
        else:
            return 'even'

    df['_count_sit'] = df.apply(get_count_situation, axis=1)

    # 상황별 구종 비율 집계
    TARGET_PITCHES = ['FF', 'SL', 'CH', 'SI', 'CU', 'FC']
    situations = ['ahead', 'behind', 'even']
    matchups = ['L', 'R']

    # 카운트 상황별 구종 비율
    for sit in situations:
        mask = df['_count_sit'] == sit
        sit_df = df[mask]
        sit_pct = (
            sit_df.groupby(['pitcher', 'game_year'])['pitch_type']
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
            .reset_index()
        )
        for pt in TARGET_PITCHES:
            col_name = f'pitcher_{pt.lower()}_pct_{sit}'
            if pt not in sit_pct.columns:
                sit_pct[pt] = 0.0
            sit_pct = sit_pct.rename(columns={pt: col_name})
            df = df.merge(
                sit_pct[['pitcher', 'game_year', col_name]],
                on=['pitcher', 'game_year'],
                how='left'
            )
            league_avg = df[col_name].mean()
            df[col_name] = df[col_name].fillna(league_avg)

    # 매치업별 구종 비율 (vs 좌타 / vs 우타)
    for hand in matchups:
        if hand == 'L':
            mask = (df['stand'] == 'L') | (df['stand'] == 1)
        else:
            mask = (df['stand'] == 'R') | (df['stand'] == 0)
        hand_df = df[mask]
        hand_pct = (
            hand_df.groupby(['pitcher', 'game_year'])['pitch_type']
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
            .reset_index()
        )
        for pt in TARGET_PITCHES:
            col_name = f'pitcher_{pt.lower()}_pct_vs{hand}'
            if pt not in hand_pct.columns:
                hand_pct[pt] = 0.0
            hand_pct = hand_pct.rename(columns={pt: col_name})
            df = df.merge(
                hand_pct[['pitcher', 'game_year', col_name]],
                on=['pitcher', 'game_year'],
                how='left'
            )
            league_avg = df[col_name].mean()
            df[col_name] = df[col_name].fillna(league_avg)

    # 임시 컬럼 제거
    df = df.drop(columns=['_count_sit'])

    n_new = len(situations) * len(TARGET_PITCHES) + len(matchups) * len(TARGET_PITCHES)
    print(f"  신규 피처 수: {n_new}개")
    print(f"  카운트 상황별: {len(situations) * len(TARGET_PITCHES)}개")
    print(f"  매치업별: {len(matchups) * len(TARGET_PITCHES)}개")
    print("Pitcher Situation 피처 엔지니어링 완료")
    return df