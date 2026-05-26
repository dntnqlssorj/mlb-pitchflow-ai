import numpy as np
import torch
from torch.utils.data import Dataset
from ml_engine.config import ALLOWED_FEATURES, TRAIN_YEAR, TEST_YEAR

SEQUENCE_LENGTH = 5

class PitchSequenceDataset(Dataset):
    """
    [3D 텐서 시퀀스 데이터셋]
    - 입력: [Batch, SEQUENCE_LENGTH, FEATURE_DIM]
    - 타겟: 현재 투구 구종 (정수 인코딩)
    - 누수 차단: 윈도우는 반드시 현재 투구 이전 N구로만 구성
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_sequence_dataset(df, label_encoder, split='train', features=None):
    """
    [슬라이딩 윈도우 3D 텐서 구축]
    - split: 'train' (2024) | 'test' (2025)
    - 반환: PitchSequenceDataset
    """
    import pandas as pd

    # - 연도 필터
    year = TRAIN_YEAR if split == 'train' else TEST_YEAR
    df_split = df[df['game_year'] == year].copy()

    # - 피처 컬럼 추출 (features 또는 ALLOWED_FEATURES 기준)
    feature_list = ALLOWED_FEATURES if features is None else features
    available = [f for f in feature_list if f in df_split.columns]
    df_split = df_split.fillna(0)

    # - 정렬 (시간 순서 보장)
    df_split = df_split.sort_values(
        ['game_pk', 'pitcher', 'at_bat_number', 'pitch_number']
    ).reset_index(drop=True)

    # - 라벨 인코딩
    df_split['pitch_encoded'] = label_encoder.transform(
        df_split['pitch_type'].astype(str)
    )

    X_list, y_list = [], []

    # - 그룹별 슬라이딩 윈도우
    for (game_pk, pitcher), group in df_split.groupby(
        ['game_pk', 'pitcher'], sort=False
    ):
        feats = group[available].values.astype(np.float32)
        labels = group['pitch_encoded'].values

        for i in range(len(group)):
            # 현재 투구 이전 SEQUENCE_LENGTH구 추출
            start = max(0, i - SEQUENCE_LENGTH)
            window = feats[start:i]  # 현재 투구 자신 미포함

            # Zero padding (앞쪽 채움)
            if len(window) < SEQUENCE_LENGTH:
                pad = np.zeros(
                    (SEQUENCE_LENGTH - len(window), len(available)),
                    dtype=np.float32
                )
                window = np.vstack([pad, window]) if len(window) > 0 else \
                         np.zeros((SEQUENCE_LENGTH, len(available)), dtype=np.float32)

            X_list.append(window)
            y_list.append(labels[i])

    X = np.array(X_list, dtype=np.float32)   # (N, 5, F)
    y = np.array(y_list, dtype=np.int64)      # (N,)

    print(f"[{split}] 텐서 shape: {X.shape}, 라벨 수: {len(np.unique(y))}")
    return PitchSequenceDataset(X, y)
