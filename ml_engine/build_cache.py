import joblib
import pandas as pd
from pathlib import Path
from ml_engine.datasets import get_clean_datasets
from ml_engine.feature_engineering import (
    build_season_baseline,
    add_pitcher_repertoire_features,
    add_pitcher_situation_features,
    add_situational_features,
)

MODEL_DIR = Path("ml_engine/models")

def build_enrichment_cache():
    datasets = get_clean_datasets()
    bat_df = datasets["bat_tracking"]
    blocking_df = datasets["blocking"]
    oaa_df = datasets["oaa"]

    # 1. 투수 베이스라인
    baseline = build_season_baseline(bat_df)
    baseline_dict = {
        (row.pitcher, row.game_year): {
            "base_speed": row.season_avg_speed,
            "base_spin": row.season_avg_spin,
            "p_throws": row.p_throws,
        }
        for _, row in baseline.iterrows()
    }
    joblib.dump(baseline_dict, MODEL_DIR / "enrichment_pitcher_baseline.pkl", compress=3)
    print(f"[완료] enrichment_pitcher_baseline.pkl — {len(baseline_dict)}개 (투수×시즌)")

    # 2. 투수 구종 비율 (기본 6개 + 상황별 30개)
    REPERTOIRE_COLS = [
        "pitcher_ff_pct", "pitcher_sl_pct", "pitcher_ch_pct",
        "pitcher_si_pct", "pitcher_cu_pct", "pitcher_fc_pct",
        "pitcher_ff_pct_ahead", "pitcher_sl_pct_ahead", "pitcher_ch_pct_ahead",
        "pitcher_si_pct_ahead", "pitcher_cu_pct_ahead", "pitcher_fc_pct_ahead",
        "pitcher_ff_pct_behind", "pitcher_sl_pct_behind", "pitcher_ch_pct_behind",
        "pitcher_si_pct_behind", "pitcher_cu_pct_behind", "pitcher_fc_pct_behind",
        "pitcher_ff_pct_even", "pitcher_sl_pct_even", "pitcher_ch_pct_even",
        "pitcher_si_pct_even", "pitcher_cu_pct_even", "pitcher_fc_pct_even",
        "pitcher_ff_pct_vsL", "pitcher_sl_pct_vsL", "pitcher_ch_pct_vsL",
        "pitcher_si_pct_vsL", "pitcher_cu_pct_vsL", "pitcher_fc_pct_vsL",
        "pitcher_ff_pct_vsR", "pitcher_sl_pct_vsR", "pitcher_ch_pct_vsR",
        "pitcher_si_pct_vsR", "pitcher_cu_pct_vsR", "pitcher_fc_pct_vsR",
    ]
    df_rep = add_pitcher_repertoire_features(bat_df.copy())
    df_rep = add_pitcher_situation_features(df_rep)
    df_rep = add_situational_features(df_rep)
    rep_agg = df_rep.groupby(["pitcher", "game_year"])[REPERTOIRE_COLS].mean().reset_index()
    rep_dict = {
        (row.pitcher, row.game_year): {col: row[col] for col in REPERTOIRE_COLS}
        for _, row in rep_agg.iterrows()
    }
    joblib.dump(rep_dict, MODEL_DIR / "enrichment_pitcher_repertoire.pkl", compress=3)
    print(f"[완료] enrichment_pitcher_repertoire.pkl — {len(rep_dict)}개 (투수×시즌)")

    # 3. 포수 블로킹
    blocking_dict = {
        (row.player_id, row.game_year): {"catcher_blocking_runs": float(row.catcher_blocking_runs)}
        for _, row in blocking_df.iterrows()
        if pd.notna(row.get("catcher_blocking_runs"))
    }
    joblib.dump(blocking_dict, MODEL_DIR / "enrichment_catcher_blocking.pkl", compress=3)
    print(f"[완료] enrichment_catcher_blocking.pkl — {len(blocking_dict)}개")

    # 4. 야수 OAA
    oaa_dict = {
        (row.player_id, row.game_year): {"outs_above_average": float(row.outs_above_average)}
        for _, row in oaa_df.iterrows()
        if pd.notna(row.get("outs_above_average"))
    }
    joblib.dump(oaa_dict, MODEL_DIR / "enrichment_fielding_oaa.pkl", compress=3)
    print(f"[완료] enrichment_fielding_oaa.pkl — {len(oaa_dict)}개")

    # 5. 투수 레퍼토리 & 평균 구속/위치 집계 캐시 (시각화 전용)
    print("\n - 투수 레퍼토리 및 평균 구속/탄착군 캐시 생성 중...")
    bat_df_clean = bat_df.dropna(subset=['pitcher', 'game_year', 'pitch_type']).copy()
    
    # step 1. groupby(['pitcher','game_year','pitch_type']).size() -> cnt
    cnt_df = bat_df_clean.groupby(['pitcher', 'game_year', 'pitch_type']).size().reset_index(name='cnt')
    
    # step 2. pct = cnt / groupby(['pitcher','game_year'])['cnt'].transform('sum') * 100
    cnt_df['pct'] = cnt_df['cnt'] / cnt_df.groupby(['pitcher', 'game_year'])['cnt'].transform('sum') * 100
    
    # step 3. groupby(['pitcher','game_year','pitch_type']).agg(...)
    agg_df = bat_df_clean.groupby(['pitcher', 'game_year', 'pitch_type']).agg(
        avg_speed=('release_speed', 'mean'),
        avg_plate_x=('plate_x', 'mean'),
        avg_plate_z=('plate_z', 'mean')
    ).round(2).reset_index()
    
    # step 4. merge step1+step2 결과와 step3 결과를 LEFT JOIN
    merged_df = pd.merge(cnt_df, agg_df, on=['pitcher', 'game_year', 'pitch_type'], how='left')
    
    # step 5. 필터: pct < 1.0 또는 cnt < 10 인 행 제거
    filtered_df = merged_df[~((merged_df['pct'] < 1.0) | (merged_df['cnt'] < 10))]
    
    # 컬럼 선정 및 강제 변환
    result_df = filtered_df[['pitcher', 'game_year', 'pitch_type', 'pct', 'avg_speed', 'avg_plate_x', 'avg_plate_z']].copy()
    result_df['pitcher'] = result_df['pitcher'].astype(int)
    result_df['game_year'] = result_df['game_year'].astype(int)
    result_df['pitch_type'] = result_df['pitch_type'].astype(str)
    
    # step 6. joblib.dump(결과, 'ml_engine/cache/pitcher_arsenal_cache.pkl')
    cache_dir = Path("ml_engine/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result_df, cache_dir / "pitcher_arsenal_cache.pkl", compress=3)
    print(f"[완료] pitcher_arsenal_cache.pkl — {len(result_df)}개 레코드 저장 완료")

    print("\n전체 enrichment 캐시 생성 완료")

if __name__ == "__main__":
    build_enrichment_cache()
