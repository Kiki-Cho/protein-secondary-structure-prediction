"""蛋白质二级结构数据读取、检查、划分与批处理。"""

from __future__ import annotations

import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: index + 1 for index, aa in enumerate(AMINO_ACIDS)}
PAD_INDEX = 0
UNK_INDEX = len(AA_TO_INDEX) + 1

LABELS = ("C", "H", "E")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
IGNORE_INDEX = -100


def find_pickle_files(raw_dir: Path) -> list[Path]:
    """递归查找原始数据，不复制或修改文件。"""
    return sorted(raw_dir.rglob("*.pkl"))


def load_sample(path: Path) -> tuple[str, str]:
    with path.open("rb") as handle:
        sample = pickle.load(handle)
    if not isinstance(sample, dict) or set(sample) != {"seq", "ssp"}:
        raise ValueError("样本必须是只包含 seq 和 ssp 的字典")
    seq, ssp = sample["seq"], sample["ssp"]
    if not isinstance(seq, str) or not isinstance(ssp, str):
        raise ValueError("seq 和 ssp 必须是字符串")
    if not seq or len(seq) != len(ssp):
        raise ValueError("seq/ssp 不能为空且长度必须一致")
    return seq, ssp


def inspect_samples(paths: Iterable[Path]) -> dict:
    """统计长度、字符、类别比例和异常样本。"""
    lengths: list[int] = []
    amino_acids: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    anomalies: list[dict[str, str]] = []

    paths = list(paths)
    for path in paths:
        try:
            seq, ssp = load_sample(path)
            unknown_aa = sorted(set(seq) - set(AMINO_ACIDS))
            unknown_labels = sorted(set(ssp) - set(LABELS))
            if unknown_aa or unknown_labels:
                anomalies.append(
                    {
                        "file": path.name,
                        "reason": f"unknown_aa={unknown_aa}, unknown_labels={unknown_labels}",
                    }
                )
            lengths.append(len(seq))
            amino_acids.update(seq)
            labels.update(ssp)
        except Exception as error:  # 汇总坏样本，不中断整批检查
            anomalies.append({"file": path.name, "reason": str(error)})

    sorted_lengths = sorted(lengths)
    total_residues = sum(labels.values())
    median = (
        sorted_lengths[len(sorted_lengths) // 2]
        if len(sorted_lengths) % 2
        else sum(sorted_lengths[len(sorted_lengths) // 2 - 1 : len(sorted_lengths) // 2 + 1]) / 2
    )
    return {
        "sample_count": len(paths),
        "valid_sample_count": len(lengths),
        "total_residues": total_residues,
        "length": {
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "median": median,
        },
        "amino_acid_counts": dict(sorted(amino_acids.items())),
        "label_counts": dict(labels),
        "label_ratios": {
            label: labels[label] / total_residues for label in LABELS
        },
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }


def split_paths(
    paths: list[Path], seed: int = 42, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> tuple[list[Path], list[Path], list[Path]]:
    """按完整蛋白质文件划分，确保同一蛋白不跨集合。"""
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


class ProteinDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq, ssp = load_sample(self.paths[index])
        encoded_seq = torch.tensor(
            [AA_TO_INDEX.get(aa, UNK_INDEX) for aa in seq], dtype=torch.long
        )
        encoded_labels = torch.tensor(
            [LABEL_TO_INDEX[label] for label in ssp], dtype=torch.long
        )
        return encoded_seq, encoded_labels


def collate_batch(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """将可变长度序列padding到当前batch最大长度，并生成有效位点mask。"""
    batch_size = len(batch)
    max_length = max(sequence.numel() for sequence, _ in batch)
    sequences = torch.full((batch_size, max_length), PAD_INDEX, dtype=torch.long)
    labels = torch.full((batch_size, max_length), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros((batch_size, max_length), dtype=torch.bool)

    for row, (sequence, target) in enumerate(batch):
        length = sequence.numel()
        sequences[row, :length] = sequence
        labels[row, :length] = target
        mask[row, :length] = True
    return {"sequence": sequences, "labels": labels, "mask": mask}
