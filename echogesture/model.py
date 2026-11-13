"""Compact temporal classifier over keypoint sequences (edge-friendly: ~1M params, ~1 MB INT8).

Only Conv1d / BatchNorm / SiLU / MatMul / Softmax ops, so it exports cleanly to ONNX and TFLite
and quantizes well (BatchNorm folds into the convs at export time).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .features import FEAT_DIM, SEQ_LEN


class DSConvBlock(nn.Module):
    """Depthwise-separable temporal conv + pointwise expansion (MobileNet-style), residual."""

    def __init__(self, dim: int, kernel: int = 5, dilation: int = 1, expand: int = 2, drop: float = 0.1):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.dw = nn.Conv1d(dim, dim, kernel, padding=pad, dilation=dilation, groups=dim, bias=False)
        self.bn1 = nn.BatchNorm1d(dim)
        self.pw1 = nn.Conv1d(dim, dim * expand, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(dim * expand)
        self.pw2 = nn.Conv1d(dim * expand, dim, 1, bias=False)
        self.bn3 = nn.BatchNorm1d(dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        y = self.act(self.bn1(self.dw(x)))
        y = self.act(self.bn2(self.pw1(y)))
        y = self.bn3(self.pw2(y))
        return x + self.drop(y)


class SignNet(nn.Module):
    def __init__(self, num_classes: int, dim: int = 256, depth: int = 6, feat_dim: int = FEAT_DIM, drop: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(feat_dim, dim, 1, bias=False), nn.BatchNorm1d(dim), nn.SiLU(), nn.Dropout(0.1)
        )
        dilations = [1, 2, 4, 1, 2, 4, 1, 2][:depth]
        self.blocks = nn.Sequential(*[DSConvBlock(dim, 5, d) for d in dilations])
        self.attn = nn.Conv1d(dim, 1, 1)  # attention pooling over time
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(dim * 2, num_classes))

    def forward(self, x):  # x: (B, T, F)
        x = self.blocks(self.stem(x.transpose(1, 2)))  # (B, D, T)
        w = torch.softmax(self.attn(x), dim=-1)  # (B, 1, T)
        att = (x * w).sum(-1)
        avg = x.mean(-1)
        return self.head(torch.cat([att, avg], dim=1))


def example_input(batch: int = 1) -> torch.Tensor:
    return torch.zeros(batch, SEQ_LEN, FEAT_DIM)
