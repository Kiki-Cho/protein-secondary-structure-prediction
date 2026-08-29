"""在固定测试集上计算Q3、逐类别指标和混淆矩阵。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import LABELS, ProteinDataset, UNK_INDEX, collate_batch, find_pickle_files, split_paths
from models import build_model

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
CHECKPOINT_DIR = ROOT / "checkpoints"
FIGURE_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def compute_metrics(confusion: np.ndarray) -> dict:
    total = int(confusion.sum())
    metrics = {"q3_accuracy": float(np.trace(confusion) / total), "classes": {}}
    for index, label in enumerate(LABELS):
        tp = int(confusion[index, index])
        support = int(confusion[index, :].sum())
        predicted = int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics["classes"][label] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support
        }
    metrics["macro_f1"] = float(
        sum(item["f1"] for item in metrics["classes"].values()) / len(LABELS)
    )
    metrics["confusion_matrix"] = confusion.tolist()
    metrics["label_order"] = list(LABELS)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mlp", "cnn"), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    checkpoint_path = CHECKPOINT_DIR / f"{args.model}_best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(args.model, UNK_INDEX + 1, len(LABELS))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _, _, test_paths = split_paths(find_pickle_files(RAW_DIR), seed=args.seed)
    loader = DataLoader(
        ProteinDataset(test_paths), batch_size=args.batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_batch,
    )
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["sequence"])
            predictions = logits.argmax(dim=-1)
            valid_targets = batch["labels"][batch["mask"]].numpy()
            valid_predictions = predictions[batch["mask"]].numpy()
            np.add.at(confusion, (valid_targets, valid_predictions), 1)

    metrics = compute_metrics(confusion)
    metrics.update({"model": args.model, "test_proteins": len(test_paths)})
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    metrics_path = RESULTS_DIR / f"{args.model}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    figure_path = FIGURE_DIR / f"{args.model}_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusion, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=LABELS, yticklabels=LABELS,
           xlabel="Predicted label", ylabel="True label", title=f"{args.model.upper()} confusion matrix")
    threshold = confusion.max() / 2
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(confusion[row, col]), ha="center", va="center",
                    color="white" if confusion[row, col] > threshold else "black")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    print(f"{args.model.upper()} test_q3={metrics['q3_accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")
    print(f"metrics={metrics_path} confusion_matrix={figure_path}")


if __name__ == "__main__":
    main()
