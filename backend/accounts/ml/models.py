import torch.nn as nn
import numpy as np


def create_sequences(data, target_idx, seq_len):

    X, y = [], []

    for i in range(len(data) - seq_len):

        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, target_idx])

    return np.array(X), np.array(y)


class LSTMModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        out, _ = self.lstm(x)

        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):

        out, _ = self.gru(x)

        return self.fc(out[:, -1, :])
