"""
피처 소거 실험 (Feature Ablation Study)

[목적]
특정 피처 그룹을 제거했을 때 모델 예측력이 얼마나 떨어지는지 시각적으로 비교합니다.

[실행 방법]
  python -m ml_engine.ablation
  python -m ml_engine.ablation --sampling 0.3
"""
import argparse
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 없는 환경 대응
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from ml_engine.datasets import get_clean_datasets
from ml_engine.feature_engineering import (
    calculate_pitcher_stamina_decay,
    integrate_catcher_blocking,
    integrate_fielding_oaa,
)

# 한글 폰트 설정 (Windows)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# ── 베이스라인 피처 (투구 전 전체) ─────────────────────────────────────────
PRE_PITCH_FEATURES = [
    "inning", "balls", "strikes", "outs_when_up",
    "on_1b", "on_2b", "on_3b", "game_year",
    "pitcher", "batter", "fielder_2",
    "fielder_3", "fielder_4", "fielder_5", "fielder_6",
    "fielder_7", "fielder_8", "fielder_9",
    "pitch_count_in_game", "stamina_index",
    "velocity_decay_ratio", "spin_decay_ratio",
    "base_speed", "base_spin",
    "is_risp", "blocking_leverage_factor", "catcher_blocking_runs",
    "team_oaa_total", "fielding_risk_index",
    "n_thruorder_pitcher", "pitcher_days_since_prev_game",
    "at_bat_number", "pitch_number",
]

# ── 소거 실험 정의 ─────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "name": "베이스라인 (전체 피처)",
        "remove": [],
        "color": "#2ecc71",
    },
    {
        "name": "투수 ID 제거\n(pitcher)",
        "remove": ["pitcher"],
        "color": "#e74c3c",
    },
    {
        "name": "볼카운트 제거\n(balls, strikes)",
        "remove": ["balls", "strikes"],
        "color": "#e67e22",
    },
    {
        "name": "투수 체력 그룹 제거\n(stamina_index 등 6개)",
        "remove": [
            "stamina_index", "pitch_count_in_game",
            "velocity_decay_ratio", "spin_decay_ratio",
            "base_speed", "base_spin",
        ],
        "color": "#9b59b6",
    },
    {
        "name": "포수·수비 도메인 제거\n(is_risp 등 5개)",
        "remove": [
            "is_risp", "blocking_leverage_factor", "catcher_blocking_runs",
            "team_oaa_total", "fielding_risk_index",
        ],
        "color": "#3498db",
    },
]


def load_data(sampling_rate: float) -> tuple:
    datasets = get_clean_datasets()
    bat_df = datasets["bat_tracking"]

    if sampling_rate < 1.0:
        bat_df = bat_df.sample(frac=sampling_rate, random_state=42).copy()

    df = calculate_pitcher_stamina_decay(bat_df, baseline_pitches=15)
    df = integrate_catcher_blocking(df, datasets["blocking"])
    df = integrate_fielding_oaa(df, datasets["oaa"])
    df = df.dropna(subset=["pitch_type"])

    valid = df["pitch_type"].value_counts()
    df = df[df["pitch_type"].isin(valid[valid >= 10].index)].copy()

    le = LabelEncoder()
    y = le.fit_transform(df["pitch_type"])
    return df, y, le


def run_experiment(df: pd.DataFrame, y: np.ndarray, remove_features: list) -> dict:
    features = [f for f in PRE_PITCH_FEATURES if f in df.columns and f not in remove_features]
    X = df[features].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=100, max_depth=10,
        random_state=42, n_jobs=-1, eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))

    return {"accuracy": acc, "n_features": len(features)}


def plot_results(results: list, output_path: Path):
    names = [r["name"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]
    colors = [r["color"] for r in results]
    baseline_acc = accs[0]
    drops = [baseline_acc - a for a in accs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("피처 소거 실험 (Feature Ablation Study)", fontsize=16, fontweight="bold", y=1.02)

    # ── 왼쪽: 절대 정확도 ────────────────────────────────────────────────
    bars = ax1.bar(range(len(names)), accs, color=colors, edgecolor="white", linewidth=1.5)
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, fontsize=9)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_title("모델별 정확도 비교", fontsize=13, fontweight="bold")
    ax1.set_ylim(max(0, min(accs) - 5), min(100, max(accs) + 3))
    ax1.axhline(y=baseline_acc, color="#2ecc71", linestyle="--", linewidth=1.2, alpha=0.7)

    for bar, acc in zip(bars, accs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{acc:.2f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # ── 오른쪽: 베이스라인 대비 하락폭 ──────────────────────────────────
    drop_colors = ["#95a5a6" if d == 0 else "#e74c3c" if d > 1 else "#f39c12" for d in drops]
    bars2 = ax2.bar(range(len(names)), drops, color=drop_colors, edgecolor="white", linewidth=1.5)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=9)
    ax2.set_ylabel("정확도 하락폭 (%p)", fontsize=11)
    ax2.set_title("베이스라인 대비 정확도 하락", fontsize=13, fontweight="bold")
    ax2.axhline(y=0, color="black", linewidth=0.8)

    for bar, drop in zip(bars2, drops):
        label = "기준" if drop == 0 else f"-{drop:.2f}%p"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            label,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n📊 차트 저장 완료: {output_path}")


def run_ablation(sampling_rate: float = 0.1):
    print("=" * 62)
    print("  MLB PitchFlow AI - 피처 소거 실험")
    print("=" * 62)
    print(f"\n[*] 데이터 로드 중 (샘플링 {sampling_rate * 100:.0f}%) ...")

    df, y, le = load_data(sampling_rate)
    print(f"  총 데이터: {len(df):,}건 | 구종 수: {len(le.classes_)}개\n")

    results = []
    for exp in EXPERIMENTS:
        removed = exp["remove"]
        label = "없음" if not removed else ", ".join(removed)
        print(f"  실험: {exp['name'].replace(chr(10), ' ')} | 제거 피처: {label}")

        result = run_experiment(df, y, removed)
        results.append({
            "name": exp["name"],
            "color": exp["color"],
            **result,
        })
        print(f"    → Accuracy: {result['accuracy'] * 100:.2f}%  (피처 수: {result['n_features']}개)")

    # 결과 요약 출력
    print("\n" + "=" * 62)
    print("  종합 결과")
    print("=" * 62)
    baseline = results[0]["accuracy"]
    print(f"  {'실험':<30} {'Accuracy':>10}  {'하락폭':>8}")
    print(f"  {'-' * 52}")
    for r in results:
        drop = baseline - r["accuracy"]
        drop_str = "기준" if drop == 0 else f"-{drop * 100:.2f}%p"
        name = r["name"].replace("\n", " ")
        print(f"  {name:<30} {r['accuracy'] * 100:>9.2f}%  {drop_str:>8}")

    output_path = Path("ablation_result.png")
    plot_results(results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling", type=float, default=0.1,
                        help="데이터 샘플링 비율 (기본값 0.1 = 10%%)")
    args = parser.parse_args()
    run_ablation(sampling_rate=args.sampling)
