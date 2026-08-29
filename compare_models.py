"""比较两种模型，只保留测试Q3更高的checkpoint。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
CHECKPOINT_DIR = ROOT / "checkpoints"


def main() -> None:
    results = {}
    for model in ("mlp", "cnn"):
        results[model] = json.loads(
            (RESULTS_DIR / f"{model}_metrics.json").read_text(encoding="utf-8")
        )
    winner = max(results, key=lambda name: results[name]["q3_accuracy"])
    loser = "cnn" if winner == "mlp" else "mlp"
    loser_checkpoint = CHECKPOINT_DIR / f"{loser}_best.pt"
    if loser_checkpoint.exists():
        loser_checkpoint.unlink()

    comparison = {
        "best_model": winner,
        "best_checkpoint": str(Path("checkpoints") / f"{winner}_best.pt"),
        "models": {
            model: {
                "q3_accuracy": values["q3_accuracy"],
                "macro_f1": values["macro_f1"],
            }
            for model, values in results.items()
        },
    }
    output = RESULTS_DIR / "model_comparison.json"
    output.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    figure = FIGURE_DIR / "model_comparison.png"
    models = ["MLP", "1D CNN"]
    scores = [results["mlp"]["q3_accuracy"], results["cnn"]["q3_accuracy"]]
    plt.figure(figsize=(6, 5))
    bars = plt.bar(models, scores, color=["#4C78A8", "#F58518"])
    plt.ylim(0, 1)
    plt.ylabel("Test Q3 accuracy")
    plt.title("Model comparison")
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2, score + 0.015, f"{score:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(figure, dpi=160)
    plt.close()
    print(f"best_model={winner} best_checkpoint={comparison['best_checkpoint']}")


if __name__ == "__main__":
    main()
