"""
Phase 4A: 1D CNN Architecture for PPG-Based Blood Pressure Estimation.

Architecture:
    Input [B, n_channels, 1250]
    → 4 ConvBlocks (Conv1D + BN + ReLU + Pooling)
    → AdaptiveAvgPool1D(1)
    → Flatten
    → FC(128→64) + ReLU + Dropout(0.2)
    → SBP head: Linear(64→1)
    → DBP head: Linear(64→1)
    → Output [B, 2]

Supports n_channels=1 (PPG only, Model A) or n_channels=3 (PPG+VPG+APG, Model B).
All other parameters are identical between models for a fair ablation.
"""

import torch
import torch.nn as nn
from typing import Tuple


class ConvBlock(nn.Module):
    """
    Single convolutional block: Conv1D → BatchNorm → ReLU → Pooling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int = 2,
        use_adaptive_pool: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding="same", bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_adaptive_pool = use_adaptive_pool
        if use_adaptive_pool:
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            self.pool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class PPGCNNBaseline(nn.Module):
    """
    Compact 1D CNN for cuffless blood pressure estimation from PPG.

    Architecture summary:
        Block 1: Conv(n_ch→32, k=7) + BN + ReLU + MaxPool(2)  → L/2
        Block 2: Conv(32→64, k=7)   + BN + ReLU + MaxPool(2)  → L/4
        Block 3: Conv(64→128, k=5)  + BN + ReLU + MaxPool(2)  → L/8
        Block 4: Conv(128→128, k=5) + BN + ReLU + AdaptiveAvgPool(1) → [B,128,1]
        Flatten → [B, 128]
        FC(128→64) + ReLU + Dropout(0.2)
        SBP head: FC(64→1)
        DBP head: FC(64→1)
        Output: [B, 2]

    Args:
        n_channels: Number of input channels. 1=PPG only, 3=PPG+VPG+APG.
        dropout:    Dropout rate in the FC layer. Default: 0.2.
    """

    def __init__(self, n_channels: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        assert n_channels in (1, 3), f"n_channels must be 1 or 3, got {n_channels}"
        self.n_channels = n_channels

        self.block1 = ConvBlock(n_channels, 32, kernel_size=7, pool_size=2)
        self.block2 = ConvBlock(32, 64, kernel_size=7, pool_size=2)
        self.block3 = ConvBlock(64, 128, kernel_size=5, pool_size=2)
        self.block4 = ConvBlock(128, 128, kernel_size=5, use_adaptive_pool=True)

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.sbp_head = nn.Linear(64, 1)
        self.dbp_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, n_channels, 1250)
        Returns:
            Output tensor of shape (B, 2) where dim 1 = [SBP_pred, DBP_pred]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flatten(x)   # (B, 128)
        x = self.fc(x)        # (B, 64)
        sbp = self.sbp_head(x)  # (B, 1)
        dbp = self.dbp_head(x)  # (B, 1)
        return torch.cat([sbp, dbp], dim=1)  # (B, 2)

    def count_parameters(self) -> Tuple[int, int]:
        """Returns (total_params, trainable_params)."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def print_summary(self) -> None:
        """Print model architecture and parameter counts."""
        total, trainable = self.count_parameters()
        print("=" * 60)
        print(f"PPGCNNBaseline (n_channels={self.n_channels})")
        print("=" * 60)
        print(self)
        print("-" * 60)
        print(f"Total parameters:     {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print("=" * 60)
