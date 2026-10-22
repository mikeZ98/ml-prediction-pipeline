"""PyTorch model definition: Conv1D stack -> stacked Bi-GRU -> linear head.

A one-to-one port of the former Keras 3 network, layer for layer, so the
architecture (and its parameter count) is unchanged by the framework move.

The input is X shaped `(rows, n_features, 1)`: the feature axis plays the role of
the sequence, so Conv1D and the GRUs run *across features*, not across time. PyTorch
wants channels before length for Conv1d and after it for a batch-first GRU, hence
the two transposes in `forward`.
"""

from __future__ import annotations

from typing import Final

import torch
from torch import nn

from mlpp.config import Loss

#: L2 factor on the GRU input kernels — Keras' `kernel_regularizer=l2(1e-3)`.
GRU_KERNEL_L2: Final = 1e-3


class CnnBiGruRegressor(nn.Module):
    """Point-wise Conv1D (64 -> 128 -> 64) feeding two bidirectional GRUs and a scalar head."""

    def __init__(self, n_features: int) -> None:
        if n_features < 1:
            raise ValueError(f"n_features must be >= 1, got {n_features}")
        super().__init__()
        self.n_features = n_features
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.gru1 = nn.GRU(64, 128, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(0.3)
        self.gru2 = nn.GRU(256, 64, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(0.3)
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(batch, n_features, 1)` -> `(batch, 1)`."""
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (batch, n_features, 64)
        h, _ = self.gru1(h)
        h = self.drop1(h)
        # The final hidden states of both directions — what Keras' Bidirectional
        # returns with return_sequences=False: forward after the last step, backward
        # after the first.
        _, h_n = self.gru2(h)
        last = torch.cat([h_n[0], h_n[1]], dim=1)  # (batch, 128)
        out: torch.Tensor = self.head(self.drop2(last))
        return out

    def l2_penalty(self) -> torch.Tensor:
        """`GRU_KERNEL_L2 * sum(w**2)` over the GRU input kernels (Keras' `kernel`).

        Keras applies `kernel_regularizer` to the input weights only — not the
        recurrent kernel or the biases — so only the `weight_ih_*` tensors count.
        """
        total = torch.zeros((), device=self.head.weight.device)
        for gru in (self.gru1, self.gru2):
            for name, weight in gru.named_parameters():
                if name.startswith("weight_ih"):
                    total = total + weight.pow(2).sum()
        return GRU_KERNEL_L2 * total


def build_model(n_features: int) -> CnnBiGruRegressor:
    """The regressor for inputs of shape `(n_features, 1)`."""
    return CnnBiGruRegressor(n_features)


def make_loss(loss: Loss) -> nn.Module:
    """Per-element loss (`reduction="none"`) so sample weights can be applied."""
    if loss == "mse":
        return nn.MSELoss(reduction="none")
    # Keras' Huber default delta is 1.0, as is PyTorch's.
    return nn.HuberLoss(reduction="none", delta=1.0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
