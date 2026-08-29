"""训练 MLP 或 1D CNN；每个epoch仅输出一行，保存最佳验证checkpoint。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import (
    AA_TO_INDEX,
    IGNORE_INDEX,
    LABELS,
    ProteinDataset,
    UNK_INDEX,
    collate_batch,
    find_pickle_files,
    split_paths,
)
from models import build_model

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
CHECKPOINT_DIR = ROOT / "checkpoints"
FIGURE_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(paths: list[Path], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        ProteinDataset(paths),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_batch,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_sites = 0

    for batch in loader:
        sequence = batch["sequence"].to(device)
        labels = batch["labels"].to(device)
        mask = batch["mask"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(sequence)
            loss = criterion(logits.reshape(-1, len(LABELS)), labels.reshape(-1))
            if training:
                loss.backward()
                optimizer.step()
        valid_sites = int(mask.sum().item())
        predictions = logits.argmax(dim=-1)
        total_correct += int(((predictions == labels) & mask).sum().item())
        total_loss += float(loss.item()) * valid_sites
        total_sites += valid_sites
    return total_loss / total_sites, total_correct / total_sites


def plot_history(history: dict[str, list[float]], model_name: str) -> Path:
    FIGURE_DIR.mkdir(exist_ok=True)
    output = FIGURE_DIR / f"{model_name}_loss.png"
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Training loss")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(f"{model_name.upper()} training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("mlp", "cnn"), required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")
    paths = find_pickle_files(RAW_DIR)
    train_paths, val_paths, _ = split_paths(paths, seed=args.seed)
    if args.smoke:
        train_paths, val_paths = train_paths[:64], val_paths[:32]
        args.epochs = min(args.epochs, 2)

    train_loader = make_loader(train_paths, args.batch_size, True)
    val_loader = make_loader(val_paths, args.batch_size, False)
    model = build_model(args.model, UNK_INDEX + 1, len(LABELS)).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {"train_loss": [], "val_loss": [], "train_q3": [], "val_q3": []}
    best_val_loss = float("inf")
    stale_epochs = 0
    checkpoint_path = CHECKPOINT_DIR / f"{args.model}_best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_q3 = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_q3 = run_epoch(model, val_loader, criterion, device)
        for key, value in (
            ("train_loss", train_loss), ("val_loss", val_loss),
            ("train_q3", train_q3), ("val_q3", val_q3),
        ):
            history[key].append(value)
        print(
            f"{args.model.upper()} epoch {epoch:02d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_q3={train_q3:.4f} val_q3={val_q3:.4f}"
        )
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            stale_epochs = 0
            if not args.smoke:
                CHECKPOINT_DIR.mkdir(exist_ok=True)
                torch.save(
                    {
                        "model_name": args.model,
                        "model_state": model.state_dict(),
                        "best_val_loss": best_val_loss,
                        "seed": args.seed,
                        "label_order": LABELS,
                    },
                    checkpoint_path,
                )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"{args.model.upper()} early stopping at epoch {epoch}")
                break

    if args.smoke:
        print(f"{args.model.upper()} smoke test passed")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    history_path = RESULTS_DIR / f"{args.model}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    figure_path = plot_history(history, args.model)
    print(f"best_checkpoint={checkpoint_path} loss_figure={figure_path}")


if __name__ == "__main__":
    main()
