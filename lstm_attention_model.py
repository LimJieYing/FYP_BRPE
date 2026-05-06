import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMEstimator(nn.Module):
    def __init__(self, n_mels, hidden_size=256, num_layers=3, bidirectional=True, dropout=0.2, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_mels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.lstm_out_dim = hidden_size * (2 if bidirectional else 1)

        # Attention layer
        self.attn = nn.Sequential(
            nn.Linear(self.lstm_out_dim, self.lstm_out_dim),
            nn.Tanh(), # tanh function introduce non-linearity  
            #can help the model learn more complex attention patterns compared to a simple linear layer.
            nn.Linear(self.lstm_out_dim, 1)
        )

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(self.lstm_out_dim, self.lstm_out_dim // 2),
            nn.ReLU(), #use ReLu for activation function for faster convergence
            nn.Linear(self.lstm_out_dim // 2, out_dim)
        )

	# x: (B, T, F), lengths: (B,) 
    def forward(self, x, lengths):
        # x: (B, T, F), lengths: (B,)
        lengths_cpu = lengths.cpu()
        packed = pack_padded_sequence(x, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)

        # Unpack to (B, T, H)
        out, out_lengths = pad_packed_sequence(packed_out, batch_first=True)

        # Move lengths to the same device as out
        out_lengths = out_lengths.to(out.device)

        # Attention scores: (B, T, 1) -> (B, T)
        scores = self.attn(out).squeeze(-1)

        # Mask padding
        max_len = out.size(1)
        mask = torch.arange(max_len, device=out.device)[None, :] < out_lengths[:, None]
        scores = scores.masked_fill(~mask, -1e9)

        # Attention weights
        weights = torch.softmax(scores, dim=1)  # (B, T)

        # Weighted sum of outputs
        context = (out * weights.unsqueeze(-1)).sum(dim=1)  # (B, H)

        return self.head(context)