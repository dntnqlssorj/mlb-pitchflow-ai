import pandas as pd
import numpy as np
import warnings
import joblib
from pathlib import Path
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

from ml_engine.datasets import get_clean_datasets
from ml_engine.feature_engineering import (
    calculate_pitcher_stamina_decay,
    integrate_catcher_blocking,
    integrate_fielding_oaa
)

def prepare_training_data(sampling_rate: float = 0.1) -> tuple:
    """
    [학습 데이터 분할 및 준비]
    - 목적: ML 벤치마크 테스트를 위한 X(피처), y(타겟) 데이터셋 전처리
    - 반환값: X_train, X_test, y_train, y_test, 피처 컬럼명 리스트, 라벨 인코더
    """
    print(f"📦 데이터 로드 및 전처리 시작 (샘플링 비율: {sampling_rate * 100}%)")
    
    # - 원본 데이터 로드: 4종 마스터 데이터 호출
    datasets = get_clean_datasets()
    bat_df = datasets['bat_tracking']
    
    # - 데이터 샘플링: 빠른 파일럿 테스트를 위한 무작위 추출
    if sampling_rate < 1.0:
        bat_df = bat_df.sample(frac=sampling_rate, random_state=42).copy()
        
    # - 도메인 피처 병합: 체력 지수, 블로킹 가중치, 수비 리스크 인덱스 추가
    df = calculate_pitcher_stamina_decay(bat_df, baseline_pitches=15)
    df = integrate_catcher_blocking(df, datasets['blocking'])
    df = integrate_fielding_oaa(df, datasets['oaa'])
    
    # - 타겟 변수 결측치 제거: 예측 불가능한 불량 데이터 필터링
    df = df.dropna(subset=['pitch_type'])
    
    # - 희귀 구종 필터링: stratify 에러 방지 및 모델 학습 안정성 확보 (샘플 수 10개 미만 제외)
    valid_pitch_types = df['pitch_type'].value_counts()
    valid_pitch_types = valid_pitch_types[valid_pitch_types >= 10].index
    df = df[df['pitch_type'].isin(valid_pitch_types)]
    
    # - 타겟 인코딩: 문자열 구종(FF, SL 등)을 숫자(0, 1 등)로 변환
    le = LabelEncoder()
    y = le.fit_transform(df['pitch_type'])
    
    # - 특성 변수 추출: 타겟 변수 및 문자열(날짜/이름 등) 제외 수치형 컬럼만 확보
    X = df.select_dtypes(include=[np.number])
    
    # - 결측치 처리: 모델 학습 에러 방지를 위해 단순 평균(0) 대치
    X = X.fillna(0)
    
    # - 데이터 분할: Train/Test 8:2 비율 (구종 비율 유지 Stratify 적용)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ 데이터 준비 완료 (학습용: {len(X_train)}건, 검증용: {len(X_test)}건, 피처 수: {len(X.columns)})")
    return X_train, X_test, y_train, y_test, X.columns.tolist(), le

def evaluate_multiple_models(X_train, X_test, y_train, y_test, feature_names):
    """
    [멀티 벤치마크 파이프라인 구동]
    - 목적: 트리 기반 3대 주류 모델 성능 상호 비교 및 우승 모델 선정
    """
    print("\n🚀 멀티 벤치마크 파이프라인 학습 시작 (RF, LightGBM, XGBoost)")
    
    # - 모델 선언: 랜덤포레스트, LGBM, XGBoost 기본 빌드 (빠른 연산 튜닝)
    models = {
        'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'LightGBM': LGBMClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, verbose=-1),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, eval_metric='mlogloss')
    }
    
    results = []
    trained_models = {}
    
    # - 순회 학습 및 예측: 선언된 3개 모델 반복 수행
    for name, model in models.items():
        print(f" - [{name}] 학습 및 예측 중...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # - 평가지표 산출: 다중 분류 환경을 고려한 가중 평균(Weighted) 스코어 계산
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        trained_models[name] = model
        
    # - 성적표 생성: F1-Score 기준 내림차순 정렬 데이터프레임
    results_df = pd.DataFrame(results).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
    
    print("\n📊 [3대 모델 성능 비교 성적표]")
    print(results_df.to_string(index=False))
    
    # - 우승 모델 선정: F1-Score 1위 모델 추출
    best_model_name = results_df.iloc[0]['Model']
    best_model = trained_models[best_model_name]
    
    # - 피처 중요도 추출: 우승 모델의 Top 15 피처 텍스트 리스트 출력
    importances = best_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(15)
    
    print(f"\n🏆 [우승 모델: {best_model_name}] 피처 중요도 (Top 15)")
    for idx, row in feat_imp_df.iterrows():
        print(f" - {row['Feature']}: {row['Importance']:.4f}")
        
    return results_df, best_model

def pack_model(model, label_encoder, model_name: str = 'xgboost_pitch_model'):
    """
    [우승 모델 패킹 (Packing)]
    - 목적: 실전 서빙을 위한 모델 아티팩트 영구 저장
    - 저장 대상: 우승 모델 객체, 라벨 인코더 객체
    """
    # - 디렉토리 생성: 모델 저장용 폴더 확인 및 자동 생성
    model_dir = Path('ml_engine/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # - 모델 저장: joblib을 이용한 우승 모델 직렬화 (압축 레벨 3 적용)
    model_path = model_dir / f'{model_name}.pkl'
    joblib.dump(model, model_path, compress=3)
    
    # - 라벨 인코더 저장: 구종 복원을 위한 인코더 직렬화
    encoder_path = model_dir / 'label_encoder.pkl'
    joblib.dump(label_encoder, encoder_path, compress=3)
    
    # - 저장 결과 검증 및 출력: 경로, 용량 확인
    model_size_kb = model_path.stat().st_size / 1024
    encoder_size_kb = encoder_path.stat().st_size / 1024
    
    print(f"\n정훈 님, XGBoost 모델 및 라벨 인코더 패킹이 성공적으로 완료되었습니다!")
    print(f" - 🏆 모델 저장 경로: {model_path} ({model_size_kb:.1f} KB)")
    print(f" - 🔑 인코더 저장 경로: {encoder_path} ({encoder_size_kb:.1f} KB)")
    print(f" - ✅ FastAPI 백엔드에서 joblib.load('{model_path}')로 즉시 로드 가능")

if __name__ == "__main__":
    # - 벤치마크 테스트 실행: 데이터 샘플링 10% 기반 빠른 검증
    X_train, X_test, y_train, y_test, feat_names, label_encoder = prepare_training_data(sampling_rate=0.1)
    
    # - 평가 파이프라인 구동
    results_table, best_model = evaluate_multiple_models(X_train, X_test, y_train, y_test, feat_names)
    
    # - 모델 패킹 실행: 우승 모델 및 라벨 인코더 영구 저장
    pack_model(best_model, label_encoder)
