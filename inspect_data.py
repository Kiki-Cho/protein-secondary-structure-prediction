"""检查全部原始pickle样本并保存紧凑统计报告。"""

import json
from pathlib import Path

from dataset import find_pickle_files, inspect_samples, split_paths

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"


def main() -> None:
    paths = find_pickle_files(RAW_DIR)
    if not paths:
        raise FileNotFoundError(f"未找到pickle文件: {RAW_DIR}")
    report = inspect_samples(paths)
    train, val, test = split_paths(paths)
    report["split"] = {"train": len(train), "validation": len(val), "test": len(test)}
    report["split_unit"] = "完整蛋白质文件"
    report["random_seed"] = 42

    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "data_summary.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"样本={report['sample_count']} 残基={report['total_residues']} "
        f"长度={report['length']['min']}-{report['length']['max']} "
        f"异常={report['anomaly_count']}"
    )
    print(f"统计结果: {output}")


if __name__ == "__main__":
    main()
