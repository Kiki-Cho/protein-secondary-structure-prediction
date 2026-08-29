"""轻量级逐残基 MLP 与 1D CNN。"""

from __future__ import annotations

import torch
from torch import nn


class ResidueMLP(nn.Module):
    """独立处理每个残基，作为不利用邻域信息的基线。"""

    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embedding(sequence))


class ResidueCNN(nn.Module):
    """用一维卷积结合局部氨基酸上下文进行逐残基分类。"""

    def __init__(self, vocab_size: int, num_classes: int, embedding_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.features = nn.Sequential(
            nn.Conv1d(embedding_dim, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Conv1d(64, num_classes, kernel_size=1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(sequence).transpose(1, 2)
        logits = self.classifier(self.features(embedded))
        return logits.transpose(1, 2)


def build_model(name: str, vocab_size: int, num_classes: int) -> nn.Module:
    if name == "mlp":
        return ResidueMLP(vocab_size, num_classes)
    if name == "cnn":
        return ResidueCNN(vocab_size, num_classes)
    raise ValueError(f"未知模型: {name}")
