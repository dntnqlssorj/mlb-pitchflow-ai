# ==============================================================================
# MLB PitchFlow AI - ML 파이프라인 전역 설정 상수
# 변경 이력: 2026-05-21 Target Leakage 제거 및 Chronological Split 전환
# ==============================================================================

from typing import List

# ------------------------------------------------------------------------------
# Chronological Split 연도 설정
# ------------------------------------------------------------------------------
TRAIN_YEAR: int = 2024   # 학습 연도: 2024 시즌 데이터 전체
TEST_YEAR: int  = 2025   # 검증 연도: 2025 시즌 데이터 전체

# ------------------------------------------------------------------------------
# 피처 엔지니어링 파라미터
# ------------------------------------------------------------------------------
LABEL_COL: str             = 'pitch_type'  # 타겟 컬럼명
STAMINA_BASELINE_PITCHES: int = 15         # stamina 계산용 rolling window 크기
ROLLING_WINDOW_SIZE: int   = 5             # 미사용 (향후 확장 대비 보존)
MIN_PITCH_TYPE_COUNT: int  = 10            # 희귀 구종 필터 하한

# ------------------------------------------------------------------------------
# P0-C: ALLOWED_FEATURES — 누수 없는 허용 피처 화이트리스트 (38개 최종 확정)
# 검토 반영 기준: 2026-05-21
#   [검토 1] home_win_exp, bat_win_exp → 드롭 (누수 방어)
#   [검토 2] n_priorpa_thisgame_player_at_bat → 허용
#   [검토 3] age_pit_legacy → 드롭 (age_pit 단독 사용)
#   [검토 4] stand → 포함 (R=0 / L=1 사전 인코딩 필수)
#   [검토 5] pitcher_days_until_next_game, batter_days_until_next_game → 드롭
# ------------------------------------------------------------------------------
ALLOWED_FEATURES: List[str] = [
    # 그룹 A: 경기 상황 (pre-pitch)
    'balls',
    'strikes',
    'outs_when_up',
    'inning',
    'on_1b',
    'on_2b',
    'on_3b',
    'home_score_diff',
    'bat_score_diff',
    'count_situation',
    'matchup_type',

    # 그룹 B: 투수 이력
    'pitcher',
    'game_year',
    'n_thruorder_pitcher',
    'pitcher_days_since_prev_game',
    'age_pit',

    # 그룹 C: 타자 이력
    'batter',
    'n_priorpa_thisgame_player_at_bat',
    'age_bat',
    'stand',

    # 그룹 D: 투수 체력 파생
    'pitch_count_in_game',
    'base_speed',
    'base_spin',
    'velocity_decay_ratio',
    'spin_decay_ratio',
    'stamina_index',

    # 그룹 E: 포수 식별자 (블로킹 파생 전면 제거, ID만 유지)
    'fielder_2',

    # 그룹 F: Pitch Sequence (신규)
    'prev_pitch_1',
    'prev_pitch_2',
    'prev_pitch_3',

    # 그룹 G: Pitcher 구종 레퍼토리 비율 (신규)
    'pitcher_ff_pct',
    'pitcher_sl_pct',
    'pitcher_ch_pct',
    'pitcher_si_pct',    # 신규
    'pitcher_cu_pct',    # 신규
    'pitcher_fc_pct',    # 신규

    # 그룹 I: 투수 카운트/매치업별 구종 비율 (신규 30개)
    # 카운트 ahead (투수 유리)
    'pitcher_ff_pct_ahead', 'pitcher_sl_pct_ahead', 'pitcher_ch_pct_ahead',
    'pitcher_si_pct_ahead', 'pitcher_cu_pct_ahead', 'pitcher_fc_pct_ahead',
    # 카운트 behind (타자 유리)
    'pitcher_ff_pct_behind', 'pitcher_sl_pct_behind', 'pitcher_ch_pct_behind',
    'pitcher_si_pct_behind', 'pitcher_cu_pct_behind', 'pitcher_fc_pct_behind',
    # 카운트 even (동등)
    'pitcher_ff_pct_even', 'pitcher_sl_pct_even', 'pitcher_ch_pct_even',
    'pitcher_si_pct_even', 'pitcher_cu_pct_even', 'pitcher_fc_pct_even',
    # vs 좌타
    'pitcher_ff_pct_vsL', 'pitcher_sl_pct_vsL', 'pitcher_ch_pct_vsL',
    'pitcher_si_pct_vsL', 'pitcher_cu_pct_vsL', 'pitcher_fc_pct_vsL',
    # vs 우타
    'pitcher_ff_pct_vsR', 'pitcher_sl_pct_vsR', 'pitcher_ch_pct_vsR',
    'pitcher_si_pct_vsR', 'pitcher_cu_pct_vsR', 'pitcher_fc_pct_vsR',
]
# ALLOWED_FEATURES 총 68개

# ------------------------------------------------------------------------------
# LEAKAGE_FEATURES — 누수 드롭 목록 (검증 및 문서화 목적 참조 상수)
# 출처: research.md §2.2 직접 누수 24개 + §2.3 결과 누수 22개 + 검토 추가 5개
# ------------------------------------------------------------------------------
LEAKAGE_FEATURES: List[str] = [
    # 직접 누수 — 릴리스 운동학 (research.md §2.2)
    'release_speed',
    'release_spin_rate',
    'release_extension',
    'release_pos_x',
    'release_pos_y',
    'release_pos_z',
    'arm_angle',
    # 직접 누수 — 공기역학 추적
    'pfx_x',
    'pfx_z',
    'plate_x',
    'plate_z',
    'vx0',
    'vy0',
    'vz0',
    'ax',
    'ay',
    'az',
    'effective_speed',
    'api_break_z_with_gravity',
    'api_break_x_arm',
    'api_break_x_batter_in',
    # 직접 누수 — 회전 특성
    'spin_axis',
    'spin_dir',
    'hyper_speed',
    # 직접 누수 — 레거시
    'spin_rate_deprecated',
    'break_angle_deprecated',
    'break_length_deprecated',
    # 결과 측정 누수 — 타격 결과 물리 (research.md §2.3)
    'bat_speed',
    'swing_length',
    'attack_angle',
    'attack_direction',
    'swing_path_tilt',
    'intercept_ball_minus_batter_pos_x_inches',
    'intercept_ball_minus_batter_pos_y_inches',
    # 결과 측정 누수 — 타구 운동학
    'launch_speed',
    'launch_angle',
    'launch_speed_angle',
    'hit_distance_sc',
    'hc_x',
    'hc_y',
    # 결과 측정 누수 — 기대 성적
    'estimated_ba_using_speedangle',
    'estimated_woba_using_speedangle',
    'estimated_slg_using_speedangle',
    # 결과 측정 누수 — Win Expectancy 변동
    'delta_home_win_exp',
    'delta_run_exp',
    'delta_pitcher_run_exp',
    # 결과 측정 누수 — 사후 스코어
    'post_away_score',
    'post_home_score',
    'post_bat_score',
    'post_fld_score',
    # 결과 측정 누수 — 결과 통계
    'woba_value',
    'woba_denom',
    'babip_value',
    'iso_value',
    # 결과 측정 누수 — 사후 라벨
    'zone',
    'hit_location',
    # 검토 결정 추가 드롭 (2026-05-21)
    'age_pit_legacy',                  # 검토 3
    'pitcher_days_until_next_game',    # 검토 5
    'batter_days_until_next_game',     # 검토 5
]
# LEAKAGE_FEATURES 총 51개 (원본 46개 + 검토 추가 5개)