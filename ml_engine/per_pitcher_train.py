import joblib
import numpy as np
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from ml_engine.train import prepare_training_data

MODEL_DIR  = Path('ml_engine/models')
LOCAL_DIR  = MODEL_DIR / 'local'
MIN_PITCHES = 300

def train_per_pitcher(sampling_rate=1.0):
    print("[*] 데이터 로드 중...")
    X_train, X_test, y_train, y_test, feat_names, global_le = \
        prepare_training_data(sampling_rate=sampling_rate)

    # - 글로벌 Optuna 최적 파라미터 로드
    global_model = joblib.load(MODEL_DIR / 'xgboost_pitch_model.pkl')
    best_params  = global_model.get_params()
    best_params.pop('n_estimators', None)

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    # - train.py에서 df 재추출 필요
    # prepare_training_data가 df를 반환하도록 수정하거나
    # 아래처럼 pitcher 컬럼을 feat_names에서 확인
    pitcher_idx = list(feat_names).index('pitcher') \
                  if 'pitcher' in feat_names else None

    if pitcher_idx is None:
        print("[ERROR] pitcher 피처가 feat_names에 없음")
        return

    pitchers_train = X_train.iloc[:, pitcher_idx].values.astype(int)
    pitchers_test  = X_test.iloc[:, pitcher_idx].values.astype(int)

    unique, counts = np.unique(pitchers_train, return_counts=True)
    qualified = unique[counts >= MIN_PITCHES]
    print(f"[*] 전체 투수: {len(unique)}명 / 로컬 모델 생성 대상: {len(qualified)}명 ({MIN_PITCHES}구 이상)")

    local_f1s   = []
    saved_count = 0

    for pid in qualified:
        # 학습 데이터
        train_mask = pitchers_train == pid
        test_mask  = pitchers_test  == pid

        X_p = X_train[train_mask]
        y_p = y_train[train_mask]

        if len(np.unique(y_p)) < 2:
            continue

        # 투수별 LabelEncoder (해당 투수 구종만)
        le_p = LabelEncoder()
        y_p_enc = le_p.fit_transform(
            global_le.inverse_transform(y_p)
        )

        # 로컬 XGBoost 학습 (글로벌 파라미터 재사용)
        model_p = XGBClassifier(
            n_estimators=300,
            max_depth=best_params.get('max_depth', 4),
            learning_rate=best_params.get('learning_rate', 0.1),
            subsample=best_params.get('subsample', 0.8),
            colsample_bytree=best_params.get('colsample_bytree', 0.8),
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric='mlogloss'
        )
        model_p.fit(X_p, y_p_enc)

        # 검증 (학습데이터에서 본 구종만 평가 대상으로 삼음)
        if test_mask.sum() >= 10:
            y_test_decoded = global_le.inverse_transform(y_test[test_mask])
            seen_mask = np.isin(y_test_decoded, le_p.classes_)
            
            if seen_mask.sum() >= 10:
                X_test_p = X_test[test_mask][seen_mask]
                y_test_p_enc = le_p.transform(y_test_decoded[seen_mask])
                y_pred_p = model_p.predict(X_test_p)
                
                f1_p = f1_score(
                    y_test_p_enc,
                    y_pred_p,
                    average='weighted',
                    zero_division=0
                )
                local_f1s.append(f1_p)

        # 저장
        joblib.dump(model_p, LOCAL_DIR / f'{pid}.pkl',    compress=3)
        joblib.dump(le_p,    LOCAL_DIR / f'{pid}_le.pkl', compress=3)
        saved_count += 1

        if saved_count % 50 == 0:
            avg = np.mean(local_f1s) if local_f1s else 0
            print(f"  [{saved_count}/{len(qualified)}] 평균 F1: {avg:.4f}")

    avg_f1 = np.mean(local_f1s) if local_f1s else 0
    print(f"\n[완료] 저장된 로컬 모델: {saved_count}개")
    print(f"[로컬 모델 평균 F1] {avg_f1*100:.2f}%")
    print(f"[글로벌 모델 F1]    42.52%")
    print(f"[상승폭 추정]       +{(avg_f1-0.4252)*100:.2f}%p")

    import shutil
    dir_size = sum(f.stat().st_size for f in LOCAL_DIR.rglob('*')) / 1024**2
    print(f"[저장 용량] {dir_size:.1f} MB")

if __name__ == '__main__':
    train_per_pitcher(sampling_rate=1.0)
