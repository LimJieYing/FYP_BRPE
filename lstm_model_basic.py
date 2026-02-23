import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMEstimator(nn.Module):
    def __init__(self, n_mels, hidden_size=256, num_layers=2, bidirectional=True, dropout=0.2, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, out_dim)
        )

    def forward(self, x, lengths):
        # x: (B, T, F), lengths: (B,)
        lengths = lengths.cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)

        # Use last layer hidden state (concat if bidirectional)
        if self.lstm.bidirectional:
            last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 2*H)
        else:
            last = h_n[-1]  # (B, H)

        return self.head(last)