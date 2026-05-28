-- ==============================================================================
-- MLB PitchFlow AI - Supabase DDL (PostgreSQL)
-- 용도: n8n 파이프라인 적재 및 FastAPI 백엔드 연동용 테이블 스키마
-- 작성자: Antigravity (Local Gemini)
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. 포수 프레이밍 (Catcher Framing) 테이블
-- ------------------------------------------------------------------------------
CREATE TABLE catcher_framing (
    player_id BIGINT NOT NULL,
    game_year INT NOT NULL,
    name VARCHAR(255),
    pitches BIGINT,
    rv_tot DOUBLE PRECISION,
    pct_tot DOUBLE PRECISION,
    rv_11 DOUBLE PRECISION,
    pct_11 DOUBLE PRECISION,
    rv_12 DOUBLE PRECISION,
    pct_12 DOUBLE PRECISION,
    rv_13 DOUBLE PRECISION,
    pct_13 DOUBLE PRECISION,
    rv_14 DOUBLE PRECISION,
    pct_14 DOUBLE PRECISION,
    rv_16 DOUBLE PRECISION,
    pct_16 DOUBLE PRECISION,
    rv_17 DOUBLE PRECISION,
    pct_17 DOUBLE PRECISION,
    rv_18 DOUBLE PRECISION,
    pct_18 DOUBLE PRECISION,
    rv_19 DOUBLE PRECISION,
    pct_19 DOUBLE PRECISION,
    
    -- 연도별 선수의 고유 식별을 위해 복합 기본키 설정
    PRIMARY KEY (player_id, game_year)
);

-- ------------------------------------------------------------------------------
-- 2. 포수 블로킹 (Catcher Blocking) 테이블
-- ------------------------------------------------------------------------------
CREATE TABLE catcher_blocking (
    player_id BIGINT NOT NULL,
    game_year INT NOT NULL,
    player_name VARCHAR(255),
    team_name VARCHAR(50),
    start_year INT,
    end_year INT,
    pitches BIGINT,
    catcher_blocking_runs BIGINT,
    blocks_above_average BIGINT,
    n_pbwp BIGINT,
    x_pbwp DOUBLE PRECISION,
    blocks_above_average_per_game DOUBLE PRECISION,
    freq_pbwp_easy DOUBLE PRECISION,
    freq_pbwp_medium DOUBLE PRECISION,
    freq_pbwp_tough DOUBLE PRECISION,
    diff_pbwp_easy DOUBLE PRECISION,
    diff_pbwp_medium DOUBLE PRECISION,
    diff_pbwp_tough DOUBLE PRECISION,

    PRIMARY KEY (player_id, game_year)
);

-- ------------------------------------------------------------------------------
-- 3. 야수 OAA (Outs Above Average) 테이블
-- ------------------------------------------------------------------------------
CREATE TABLE fielding_oaa (
    player_id BIGINT NOT NULL,
    game_year INT NOT NULL,
    "last_name, first_name" VARCHAR(255),
    display_team_name VARCHAR(50),
    year INT,
    primary_pos_formatted VARCHAR(10),
    fielding_runs_prevented BIGINT,
    outs_above_average BIGINT,
    outs_above_average_infront BIGINT,
    outs_above_average_lateral_toward3bline BIGINT,
    outs_above_average_lateral_toward1bline BIGINT,
    outs_above_average_behind BIGINT,
    outs_above_average_rhh BIGINT,
    outs_above_average_lhh BIGINT,
    -- 전처리 단계에서 정수(INT)로 변환된 퍼센트 수치 반영
    actual_success_rate_formatted INT,
    adj_estimated_success_rate_formatted INT,
    diff_success_rate_formatted INT,

    PRIMARY KEY (player_id, game_year)
);

-- ------------------------------------------------------------------------------
-- 4. 타구 추적 (Statcast Bat Tracking) 대용량 테이블
-- ------------------------------------------------------------------------------
CREATE TABLE statcast_bat_tracking (
    -- 기본 정보
    game_year INT NOT NULL,
    pitch_type VARCHAR(10),
    game_date DATE,
    player_name VARCHAR(255),
    batter BIGINT NOT NULL,
    pitcher BIGINT NOT NULL,
    events VARCHAR(50),
    description VARCHAR(255),
    game_type VARCHAR(5),
    stand VARCHAR(5),
    p_throws VARCHAR(5),
    home_team VARCHAR(10),
    away_team VARCHAR(10),
    type VARCHAR(10),
    bb_type VARCHAR(50),
    des TEXT,
    
    -- 피칭 & 타격 지표 (연산 최적화를 위해 DOUBLE PRECISION 및 BIGINT)
    release_speed DOUBLE PRECISION,
    release_pos_x DOUBLE PRECISION,
    release_pos_z DOUBLE PRECISION,
    spin_dir DOUBLE PRECISION,
    spin_rate_deprecated DOUBLE PRECISION,
    break_angle_deprecated DOUBLE PRECISION,
    break_length_deprecated DOUBLE PRECISION,
    zone BIGINT,
    hit_location DOUBLE PRECISION,
    balls BIGINT,
    strikes BIGINT,
    pfx_x DOUBLE PRECISION,
    pfx_z DOUBLE PRECISION,
    plate_x DOUBLE PRECISION,
    plate_z DOUBLE PRECISION,
    
    -- 주자 상황 (Null을 0으로 전처리함 -> 정수형 매핑)
    on_3b BIGINT,
    on_2b BIGINT,
    on_1b BIGINT,
    outs_when_up BIGINT,
    inning BIGINT,
    inning_topbot VARCHAR(10),
    
    -- 추가 위치 및 물리 지표
    hc_x DOUBLE PRECISION,
    hc_y DOUBLE PRECISION,
    tfs_deprecated DOUBLE PRECISION,
    tfs_zulu_deprecated DOUBLE PRECISION,
    umpire DOUBLE PRECISION,
    sv_id DOUBLE PRECISION,
    vx0 DOUBLE PRECISION,
    vy0 DOUBLE PRECISION,
    vz0 DOUBLE PRECISION,
    ax DOUBLE PRECISION,
    ay DOUBLE PRECISION,
    az DOUBLE PRECISION,
    sz_top DOUBLE PRECISION,
    sz_bot DOUBLE PRECISION,
    hit_distance_sc DOUBLE PRECISION,
    launch_speed DOUBLE PRECISION,
    launch_angle DOUBLE PRECISION,
    effective_speed DOUBLE PRECISION,
    release_spin_rate BIGINT,
    release_extension DOUBLE PRECISION,
    game_pk BIGINT,
    
    -- 수비수 ID
    fielder_2 BIGINT NOT NULL,  -- 포수
    fielder_3 BIGINT,
    fielder_4 BIGINT,
    fielder_5 BIGINT,
    fielder_6 BIGINT,
    fielder_7 BIGINT,
    fielder_8 BIGINT,
    fielder_9 BIGINT,
    
    release_pos_y DOUBLE PRECISION,
    estimated_ba_using_speedangle DOUBLE PRECISION,
    estimated_woba_using_speedangle DOUBLE PRECISION,
    woba_value DOUBLE PRECISION,
    woba_denom DOUBLE PRECISION,
    babip_value DOUBLE PRECISION,
    iso_value DOUBLE PRECISION,
    launch_speed_angle DOUBLE PRECISION,
    at_bat_number BIGINT,
    pitch_number BIGINT,
    pitch_name VARCHAR(50),
    home_score BIGINT,
    away_score BIGINT,
    bat_score BIGINT,
    fld_score BIGINT,
    post_away_score BIGINT,
    post_home_score BIGINT,
    post_bat_score BIGINT,
    post_fld_score BIGINT,
    if_fielding_alignment VARCHAR(50),
    of_fielding_alignment VARCHAR(50),
    spin_axis BIGINT,
    delta_home_win_exp DOUBLE PRECISION,
    delta_run_exp DOUBLE PRECISION,
    bat_speed DOUBLE PRECISION,
    swing_length DOUBLE PRECISION,
    estimated_slg_using_speedangle DOUBLE PRECISION,
    delta_pitcher_run_exp DOUBLE PRECISION,
    hyper_speed DOUBLE PRECISION,
    home_score_diff BIGINT,
    bat_score_diff BIGINT,
    home_win_exp DOUBLE PRECISION,
    bat_win_exp DOUBLE PRECISION,
    age_pit_legacy BIGINT,
    age_bat_legacy BIGINT,
    age_pit BIGINT,
    age_bat BIGINT,
    n_thruorder_pitcher BIGINT,
    n_priorpa_thisgame_player_at_bat BIGINT,
    pitcher_days_since_prev_game BIGINT,
    batter_days_since_prev_game BIGINT,
    pitcher_days_until_next_game DOUBLE PRECISION,
    batter_days_until_next_game DOUBLE PRECISION,
    api_break_z_with_gravity DOUBLE PRECISION,
    api_break_x_arm DOUBLE PRECISION,
    api_break_x_batter_in DOUBLE PRECISION,
    arm_angle DOUBLE PRECISION,
    attack_angle DOUBLE PRECISION,
    attack_direction DOUBLE PRECISION,
    swing_path_tilt DOUBLE PRECISION,
    intercept_ball_minus_batter_pos_x_inches DOUBLE PRECISION,
    intercept_ball_minus_batter_pos_y_inches DOUBLE PRECISION,
    
    -- [2026-05-21 추가된 신규 고급 피처 및 파생 변수 컬럼군]
    pitch_count_in_game BIGINT,
    count_situation BIGINT,
    matchup_type BIGINT,
    base_speed DOUBLE PRECISION,
    base_spin DOUBLE PRECISION,
    velocity_decay_ratio DOUBLE PRECISION,
    spin_decay_ratio DOUBLE PRECISION,
    stamina_index DOUBLE PRECISION,
    prev_pitch_1 BIGINT,
    prev_pitch_2 BIGINT,
    prev_pitch_3 BIGINT,
    pitcher_ff_pct DOUBLE PRECISION,
    pitcher_sl_pct DOUBLE PRECISION,
    pitcher_ch_pct DOUBLE PRECISION,
    pitcher_si_pct DOUBLE PRECISION,
    pitcher_cu_pct DOUBLE PRECISION,
    pitcher_fc_pct DOUBLE PRECISION,
    
    -- 카운트 ahead (투수 유리)
    pitcher_ff_pct_ahead DOUBLE PRECISION, pitcher_sl_pct_ahead DOUBLE PRECISION, pitcher_ch_pct_ahead DOUBLE PRECISION,
    pitcher_si_pct_ahead DOUBLE PRECISION, pitcher_cu_pct_ahead DOUBLE PRECISION, pitcher_fc_pct_ahead DOUBLE PRECISION,
    -- 카운트 behind (타자 유리)
    pitcher_ff_pct_behind DOUBLE PRECISION, pitcher_sl_pct_behind DOUBLE PRECISION, pitcher_ch_pct_behind DOUBLE PRECISION,
    pitcher_si_pct_behind DOUBLE PRECISION, pitcher_cu_pct_behind DOUBLE PRECISION, pitcher_fc_pct_behind DOUBLE PRECISION,
    -- 카운트 even (동등)
    pitcher_ff_pct_even DOUBLE PRECISION, pitcher_sl_pct_even DOUBLE PRECISION, pitcher_ch_pct_even DOUBLE PRECISION,
    pitcher_si_pct_even DOUBLE PRECISION, pitcher_cu_pct_even DOUBLE PRECISION, pitcher_fc_pct_even DOUBLE PRECISION,
    -- vs 좌타 (matchup vs L)
    pitcher_ff_pct_vsL DOUBLE PRECISION, pitcher_sl_pct_vsL DOUBLE PRECISION, pitcher_ch_pct_vsL DOUBLE PRECISION,
    pitcher_si_pct_vsL DOUBLE PRECISION, pitcher_cu_pct_vsL DOUBLE PRECISION, pitcher_fc_pct_vsL DOUBLE PRECISION,
    -- vs 우타 (matchup vs R)
    pitcher_ff_pct_vsR DOUBLE PRECISION, pitcher_sl_pct_vsR DOUBLE PRECISION, pitcher_ch_pct_vsR DOUBLE PRECISION,
    pitcher_si_pct_vsR DOUBLE PRECISION, pitcher_cu_pct_vsR DOUBLE PRECISION,    pitcher_fc_pct_vsR DOUBLE PRECISION,

    -- PK: 투구 단위 고유 식별자 복합키 (plan.md §6.1)
    PRIMARY KEY (game_pk, at_bat_number, pitch_number)
);


-- ==============================================================================
-- 대용량 조회를 위한 성능 최적화 (Indexes)
-- ==============================================================================

-- 타구 추적 테이블 (Statcast Bat Tracking) 조회를 극대화하는 복합 인덱스
-- 투수(pitcher) 기준 연도별 필터링 속도 최적화
CREATE INDEX idx_bat_tracking_pitcher_year ON statcast_bat_tracking (pitcher, game_year);

-- 타자(batter) 기준 연도별 필터링 속도 최적화
CREATE INDEX idx_bat_tracking_batter_year ON statcast_bat_tracking (batter, game_year);

-- 포수(fielder_2) 기준 연도별 필터링 속도 최적화 (프레이밍/블로킹 연동용)
CREATE INDEX idx_bat_tracking_catcher_year ON statcast_bat_tracking (fielder_2, game_year);

-- 게임 날짜별 또는 경기별 조회가 잦을 경우를 대비한 인덱스
CREATE INDEX idx_bat_tracking_game_pk ON statcast_bat_tracking (game_pk);
CREATE INDEX idx_bat_tracking_game_date ON statcast_bat_tracking (game_date);

-- ==============================================================================
-- plan.md §6.2~6.3 — Covering Index 및 Sequence Index 추가
-- ==============================================================================

-- [Covering Index 1] 투수 시즌 베이스라인 조회 최적화
-- fetch_pitcher_season_baseline(): release_speed, release_spin_rate AVG 산출 시
-- Index-only scan 달성 → heap fetch 배제 → 응답 5ms → 1~2ms
CREATE INDEX idx_bat_tracking_pitcher_year_covering
ON statcast_bat_tracking (pitcher, game_year)
INCLUDE (release_speed, release_spin_rate);

-- [Covering Index 2] 포수 기반 Covering Index
-- 포수별 리드 패턴 분석 향후 확장 대비
CREATE INDEX idx_bat_tracking_catcher_year_covering
ON statcast_bat_tracking (fielder_2, game_year)
INCLUDE (release_speed, release_spin_rate);

-- [Sequence Index] 현 경기 투구 시퀀스 조회 최적화
-- fetch_pitch_count_in_game(): pitcher + game_pk COUNT(*) 쿼리
-- 직전 N구 시퀀스 ORDER BY at_bat_number, pitch_number 쿼리
-- PK는 game_pk leftmost이므로 pitcher 필터 포함 시 별도 인덱스 필요
CREATE INDEX idx_bat_tracking_sequence
ON statcast_bat_tracking (pitcher, game_pk, at_bat_number, pitch_number);
