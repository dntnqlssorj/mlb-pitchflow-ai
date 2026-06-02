import pandas as pd
from ml_engine.datasets import get_clean_datasets
from ml_engine.feature_engineering import build_season_baseline
datasets = get_clean_datasets()
bat_df = datasets['bat_tracking']
bat_df['game_date'] = pd.to_datetime(bat_df['game_date'])
season_baseline_df = build_season_baseline(bat_df)
print("Columns in season_baseline_df:")
print(season_baseline_df.columns.tolist())
