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

    print("\n전체 enrichment 캐시 생성 완료")

if __name__ == "__main__":
    build_enrichment_cache()
