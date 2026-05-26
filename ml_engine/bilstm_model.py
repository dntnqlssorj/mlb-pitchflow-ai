import torch
import torch.nn as nn

class PitchBiLSTM(nn.Module):
    """
    [Bi-LSTM 구종 예측 모델]
    - 입력: (Batch, SEQUENCE_LENGTH, FEATURE_DIM)
    - 출력: (Batch, N_CLASS) 로짓
    """
    def __init__(self, feature_dim: int, n_classes: int,
                 hidden_size: int = 128, num_layers: int = 2,
                 dropout: float = 0.3):
        super().__init__()

        self.bilstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )

        self.hidden_size = hidden_size
        lstm_out_dim = hidden_size * 2  # 양방향

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        # x: (Batch, Seq, Feature)
        lstm_out, _ = self.bilstm(x)
        # forward 방향은 마지막 timestep, backward 방향은 첫 번째 timestep 추출
        out_forward = lstm_out[:, -1, :self.hidden_size]
        out_backward = lstm_out[:, 0, self.hidden_size:]
        last = torch.cat([out_forward, out_backward], dim=-1)
        return self.classifier(last)
