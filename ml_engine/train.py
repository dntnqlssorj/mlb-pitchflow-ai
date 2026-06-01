# ==============================================================================
# MLB PitchFlow AI - ML 학습 파이프라인
# 변경 이력:
#   2026-05-21 Target Leakage 제거 및 Chronological Split 전환
#   2026-05-31 2025 데이터 학습 포함 — 날짜 기반 split 전환
#     - TRAIN_YEAR/TEST_YEAR 연도 기반 mask → game_date 날짜 기반 mask로 교체
#     - 학습: ~2025-08-31 / 검증: 2025-09-01~
# ==============================================================================

import pandas as pd
import numpy as np
import warnings
import joblib
from pathlib import Path
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

from ml_engine.datasets import get_clean_datasets
from ml_engine.feature_engineering import (
    build_season_baseline,
    calculate_pitcher_stamina_decay,
    integrate_catcher_blocking,
    integrate_fielding_oaa,
    add_pitch_sequence_features,
    add_sequence_combo_features,
    add_pitcher_repertoire_features,
    add_situational_features,
    add_leverage_features,
    add_pitcher_situation_features,
    add_batter_swing_tendency_features,
    add_interaction_features,
    add_release_pos_features,
)
from ml_engine.config import (
    ALLOWED_FEATURES,
    LEAKAGE_FEATURES,
    LABEL_COL,
    TRAIN_END_DATE,
    TEST_START_DATE,
    STAMINA_BASELINE_PITCHES,
    MIN_PITCH_TYPE_COUNT,
)


def prepare_training_data(sampling_rate: float = 0.1, return_df: bool = False) -> tuple:
    """
    [학습 데이터 분할 및 준비 — 날짜 기반 split 버전]
    - 목적: Target Leakage 없는 X(피처), y(타겟) 데이터셋 전처리
    - 반환값: X_train, X_test, y_train, y_test, 피처 컬럼명 리스트, 라벨 인코더

    변경 (2026-05-31):
        - Chronological Split: game_year 연도 기반 → game_date 날짜 기반
        - 학습: game_date <= TRAIN_END_DATE (2024 전체 + 2025 전반부)
        - 검증: game_date >= TEST_START_DATE (2025 후반부 9~10월)
    """
    print(f"데이터 로드 및 전처리 시작 (샘플링 비율: {sampling_rate * 100:.0f}%)")
    print(f"  학습 기간: ~ {TRAIN_END_DATE}")
    print(f"  검증 기간: {TEST_START_DATE} ~")

    # ------------------------------------------------------------------
    # 단계 1. 원본 데이터 로드
    # ------------------------------------------------------------------
    datasets = get_clean_datasets()
    bat_df   = datasets['bat_tracking']

    # ------------------------------------------------------------------
    # 단계 2. game_date 컬럼 datetime 변환 (split 기준)
    # ------------------------------------------------------------------
    bat_df['game_date'] = pd.to_datetime(bat_df['game_date'])

    # ------------------------------------------------------------------
    # 단계 3. 시즌 베이스라인 사전 산출 (반드시 샘플링 이전)
    # ------------------------------------------------------------------
    season_baseline_df = build_season_baseline(bat_df)

    # ------------------------------------------------------------------
    # 단계 4. 데이터 샘플링 (파일럿 모드)
    # ------------------------------------------------------------------
    if sampling_rate < 1.0:
        bat_df = bat_df.sample(frac=sampling_rate, random_state=42).copy()
        print(f"샘플링 완료: {len(bat_df):,}행")

    # ------------------------------------------------------------------
    # 단계 5. 피처 엔지니어링 체인
    # ------------------------------------------------------------------
    df = calculate_pitcher_stamina_decay(
        bat_df,
        season_baseline_df,
        baseline_pitches=STAMINA_BASELINE_PITCHES,
    )
    df = integrate_catcher_blocking(df, datasets['blocking'])
    df = integrate_fielding_oaa(df, datasets['oaa'])
    df = add_pitch_sequence_features(df)
    df = add_sequence_combo_features(df)
    df = add_pitcher_repertoire_features(df)
    df = add_situational_features(df)
    df = add_leverage_features(df)
    df = add_pitcher_situation_features(df)
    df = add_batter_swing_tendency_features(df)
    df = add_interaction_features(df)
    df = add_release_pos_features(df, season_baseline_df)

    # game_date 재변환 (피처 엔지니어링 체인 후 타입 유지 보장)
    df['game_date'] = pd.to_datetime(df['game_date'])

    # ------------------------------------------------------------------
    # 단계 6. 타겟 결측치 제거
    # ------------------------------------------------------------------
    df = df.dropna(subset=[LABEL_COL])

    # ------------------------------------------------------------------
    # 단계 7. 희귀 구종 필터링 및 Class Merging
    # ------------------------------------------------------------------
    pitch_counts = df['pitch_type'].value_counts()
    print("\n[구종별 행 수 집계]")
    for pitch, count in pitch_counts.items():
        print(f"  {pitch}: {count}행")

    RARE_THRESHOLD = 100
    rare_pitches = pitch_counts[pitch_counts < RARE_THRESHOLD].index.tolist()

    if rare_pitches:
        print(f"\n[OT 편입 대상] 임계값 {RARE_THRESHOLD}행 미만 구종: {rare_pitches}")
        df['pitch_type'] = df['pitch_type'].apply(
            lambda x: 'OT' if x in rare_pitches else x
        )
    else:
        print("\n[Class Merging 불필요] 전체 클래스 유지.")

    # 학습 기간에 없고 검증 기간에만 있는 구종 제거
    train_classes = df[df['game_date'] <= TRAIN_END_DATE]['pitch_type'].unique()
    test_only_classes = [c for c in df['pitch_type'].unique() if c not in train_classes]

    if test_only_classes:
        print(f"\n[학습 데이터 미존재 구종 제거] {test_only_classes}")
        df = df[~df['pitch_type'].isin(test_only_classes)].copy()

    print(f"\n[최종 클래스 수] {df['pitch_type'].nunique()}개")

    # ------------------------------------------------------------------
    # 단계 8. 타겟 라벨 인코딩
    # ------------------------------------------------------------------
    le = LabelEncoder()
    y  = le.fit_transform(df[LABEL_COL])

    # ------------------------------------------------------------------
    # 단계 9. stand 컬럼 수치 인코딩 (R=0, L=1)
    # ------------------------------------------------------------------
    if 'stand' in df.columns:
        df['stand'] = df['stand'].map({'R': 0, 'L': 1}).fillna(0).astype(int)

    # ------------------------------------------------------------------
    # 단계 10. 피처 화이트리스트 적용
    # ------------------------------------------------------------------
    available_features = [f for f in ALLOWED_FEATURES if f in df.columns]
    skipped = [f for f in ALLOWED_FEATURES if f not in df.columns]
    if skipped:
        print(f"[경고] 스킵된 허용 피처: {skipped}")

    leakage_in_X = [f for f in available_features if f in LEAKAGE_FEATURES]
    if leakage_in_X:
        raise RuntimeError(
            f"[치명적 오류] ALLOWED_FEATURES에 누수 컬럼 포함: {leakage_in_X}"
        )

    X = df[available_features].copy()
    print(f"최종 사용 피처 수: {len(available_features)}개")

    # ------------------------------------------------------------------
    # 단계 11. 결측치 처리
    # ------------------------------------------------------------------
    X = X.fillna(0)

    # ------------------------------------------------------------------
    # 단계 12. Chronological Split — 날짜 기반 (핵심 변경)
    # ------------------------------------------------------------------
    train_mask = (df['game_date'] <= TRAIN_END_DATE).values
    test_mask  = (df['game_date'] >= TEST_START_DATE).values

    X_train = X[train_mask]
    X_test  = X[test_mask]
    y_train = y[train_mask]
    y_test  = y[test_mask]

    if len(X_train) == 0:
        raise RuntimeError(
            f"[치명적 오류] 학습 데이터 0건. TRAIN_END_DATE={TRAIN_END_DATE} 확인 필요."
        )
    if len(X_test) == 0:
        raise RuntimeError(
            f"[치명적 오류] 검증 데이터 0건. TEST_START_DATE={TEST_START_DATE} 확인 필요."
        )

    print(f"\n[날짜 기반 Chronological Split 결과]")
    print(f"  학습 (~ {TRAIN_END_DATE}): {len(X_train):,}행")
    print(f"  검증 ({TEST_START_DATE} ~): {len(X_test):,}행")
    print(f"  피처 수: {len(available_features)}개")
    print(f"  학습 구종 분포:\n{pd.Series(y_train).value_counts().to_string()}")
    print(f"  검증 구종 분포:\n{pd.Series(y_test).value_counts().to_string()}")

    if return_df:
        return X_train, X_test, y_train, y_test, available_features, le, df
    return X_train, X_test, y_train, y_test, available_features, le


def evaluate_multiple_models(
    X_train, X_test, y_train, y_test, feature_names: list
):
    """
    [멀티 벤치마크 파이프라인 — 변경 없음]
    """
    print("\n멀티 벤치마크 파이프라인 학습 시작 (RF, LightGBM, XGBoost)")

    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, verbose=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1,
            eval_metric='mlogloss'
        ),
    }

    results        = []
    trained_models = {}

    for name, model in models.items():
        print(f"  [{name}] 학습 및 예측 중...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred,    average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred,        average='weighted', zero_division=0)

        results.append({'Model': name, 'Accuracy': acc, 'Precision': prec,
                        'Recall': rec, 'F1-Score': f1})
        trained_models[name] = model

    results_df = (
        pd.DataFrame(results)
        .sort_values(by='F1-Score', ascending=False)
        .reset_index(drop=True)
    )

    print("\n[3대 모델 성능 비교]")
    print(results_df.to_string(index=False))

    best_model_name = results_df.iloc[0]['Model']
    best_model      = trained_models[best_model_name]

    importances = best_model.feature_importances_
    feat_imp_df = (
        pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        .sort_values(by='Importance', ascending=False)
        .head(15)
    )

    print(f"\n[우승 모델: {best_model_name}] 피처 중요도 (Top 15)")
    for _, row in feat_imp_df.iterrows():
        print(f"  {row['Feature']}: {row['Importance']:.4f}")

    return results_df, best_model


def pack_model(model, label_encoder, model_name: str = 'xgboost_pitch_model'):
    """
    [우승 모델 패킹 — 변경 없음]
    """
    model_dir = Path('ml_engine/models')
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path   = model_dir / f'{model_name}.pkl'
    encoder_path = model_dir / 'label_encoder.pkl'

    joblib.dump(model,         model_path,   compress=3)
    joblib.dump(label_encoder, encoder_path, compress=3)

    model_size_kb   = model_path.stat().st_size   / 1024
    encoder_size_kb = encoder_path.stat().st_size / 1024

    print(f"\n모델 패킹 완료")
    print(f"  모델 저장 경로:   {model_path} ({model_size_kb:.1f} KB)")
    print(f"  인코더 저장 경로: {encoder_path} ({encoder_size_kb:.1f} KB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampling', type=float, default=1.0, help='데이터 샘플링 비율 (기본값: 1.0)')
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, feat_names, label_encoder = prepare_training_data(
        sampling_rate=args.sampling
    )
    results_table, best_model = evaluate_multiple_models(
        X_train, X_test, y_train, y_test, feat_names
    )
    pack_model(best_model, label_encoder)