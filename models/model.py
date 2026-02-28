"""
models/model.py — BiLSTM + Multi-Head Temporal Attention
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_size:  int   = 64
    hidden_size: int   = 128
    num_layers:  int   = 2
    num_heads:   int   = 4
    dropout:     float = 0.3
    attn_drop:   float = 0.1


class TemporalAttention(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        out, w = self.attn(x, x, x)
        return self.norm(x + out), w


class AegisLSTM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.input_size, hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers, batch_first=True, bidirectional=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        h = cfg.hidden_size * 2  # bidirectional
        self.attention  = TemporalAttention(h, cfg.num_heads, cfg.attn_drop)
        self.classifier = nn.Sequential(
            nn.Linear(h, 64), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(64, 1)
        )

    def forward(self, x):
        lstm_out, _       = self.lstm(x)
        attn_out, weights = self.attention(lstm_out)
        pooled            = attn_out.mean(dim=1)
        prob              = torch.sigmoid(self.classifier(pooled).squeeze(-1))
        return prob, weights


def build_model(input_size: int, device: str = "cpu") -> AegisLSTM:
    cfg   = ModelConfig(input_size=input_size)
    model = AegisLSTM(cfg).to(device)
    n     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] AegisLSTM | features:{input_size} | params:{n:,} | device:{device}")
    return model
