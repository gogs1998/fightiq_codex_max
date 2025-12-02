"""
Siamese sequence model (PyTorch) for winner prediction using per-fight histories.
This is a reference implementation; not executed by default.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from train_fightiq_model_golden import _build_stance_encoder, _parse_finish_time_to_seconds, DATA_DIR


@dataclass
class FightExample:
    f1_seq: np.ndarray
    f2_seq: np.ndarray
    label: float


class FightHistoryDataset(Dataset):
    def __init__(self, examples: List[FightExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return (
            torch.tensor(ex.f1_seq, dtype=torch.float32),
            torch.tensor(ex.f2_seq, dtype=torch.float32),
            torch.tensor(ex.label, dtype=torch.float32),
        )


class SiameseEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        _, h = self.encoder(x)
        return h.squeeze(0)


class SiameseModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.enc = SiameseEncoder(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, f1_seq, f2_seq):
        h1 = self.enc(f1_seq)
        h2 = self.enc(f2_seq)
        x = torch.cat([h1, h2], dim=-1)
        logits = self.head(x)
        return logits


def collate_fn(batch):
    f1, f2, y = zip(*batch)
    f1 = nn.utils.rnn.pad_sequence(f1, batch_first=True)
    f2 = nn.utils.rnn.pad_sequence(f2, batch_first=True)
    return f1, f2, torch.stack(y)


# Note: building sequences requires constructing per-fight feature vectors in chronological order per fighter.
# This is a skeleton; integrate with a precomputed per-fight feature table and use only past fights for each example.
