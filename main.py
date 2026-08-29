"""项目入口提示；完整流程请按 README 中的命令运行。"""

from inspect_data import main as inspect_data


if __name__ == "__main__":
    inspect_data()
    print("下一步请运行: python train.py --model mlp")
    print("完整复现顺序见 README.md")
