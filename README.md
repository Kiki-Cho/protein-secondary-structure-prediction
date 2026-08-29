# 蛋白质二级结构预测（MLP 与 1D CNN）

本项目针对氨基酸序列进行逐残基三分类，预测 `H`（α-螺旋）、`E`（β-折叠）或 `C`（coil/loop）。项目使用 PyTorch 实现轻量 MLP 基线和 1D CNN，并在完整蛋白质级别划分训练、验证、测试集，避免同一蛋白跨集合造成数据泄漏。

## 目录

```text
data/raw/assignment_data/  # 3000个原始pickle；只读，不复制
checkpoints/               # 最终仅保留测试Q3较高模型的最佳验证checkpoint
figures/                   # 最终loss、混淆矩阵和模型对比图
results/                   # 数据统计、训练history和评价指标
dataset.py                 # 数据检查、编码、划分、padding与mask
models.py                  # MLP和1D CNN
inspect_data.py            # 全量数据检查
train.py                   # 训练与early stopping
evaluate.py                # 测试集评价
compare_models.py          # 模型对比并清理较差checkpoint
requirements.txt
```

## 数据处理

- 每个 pickle 是一个完整蛋白质，包含等长字符串 `seq` 和 `ssp`。
- 20种标准氨基酸编码为整数，0留作padding，另设未知字符索引。
- 标签顺序固定为 `C/H/E`，padding标签设为 `-100`，由交叉熵忽略。
- `collate_batch` 仅padding到当前batch最长序列，同时生成布尔mask。
- 固定随机种子42，按蛋白质文件70%/15%/15%划分，共2100/450/450个蛋白质。

## 模型与训练

- **MLP**：32维残基embedding，经64维隐藏层独立预测每个残基；它不利用邻域上下文，是基线模型。
- **1D CNN**：embedding后使用核大小7和5的卷积提取局部序列上下文，再逐位分类。
- 损失函数为带 `ignore_index=-100` 的交叉熵；优化器为Adam，学习率0.001。
- 默认CPU、batch size 32、最多20 epochs、验证loss连续4轮不改善则early stopping。
- 每个模型训练时仅覆盖保存验证loss最优checkpoint；最终只保留测试Q3较高模型的checkpoint。

## 复现

在项目根目录执行（Windows PowerShell）：

```powershell
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt
.\.venv\Scripts\python.exe inspect_data.py
.\.venv\Scripts\python.exe train.py --model mlp --smoke --epochs 2
.\.venv\Scripts\python.exe train.py --model cnn --smoke --epochs 2
.\.venv\Scripts\python.exe train.py --model mlp
.\.venv\Scripts\python.exe evaluate.py --model mlp
.\.venv\Scripts\python.exe train.py --model cnn
.\.venv\Scripts\python.exe evaluate.py --model cnn
.\.venv\Scripts\python.exe compare_models.py
```

## 评价指标

Q3 accuracy 是所有有效残基中预测正确的比例。项目还报告每一类的 precision、recall、F1和support，以及macro-F1和混淆矩阵。所有padding位点均通过mask排除。

## 实验结果

固定随机种子42后的实测结果如下。测试集含450个完整蛋白质、66,887个有效残基。

| 模型 | 测试Q3 accuracy | Macro-F1 |
|---|---:|---:|
| MLP | 0.4876 | 0.4683 |
| 1D CNN | **0.6798** | **0.6659** |

CNN逐类别结果：

| 标签 | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| C | 0.6708 | 0.7510 | 0.7086 | 27,096 |
| H | 0.6997 | 0.7076 | 0.7036 | 23,359 |
| E | 0.6647 | 0.5231 | 0.5854 | 16,432 |

MLP在第9轮early stopping；CNN训练20轮，最佳验证loss为第20轮的0.7634。CNN比MLP的Q3提高约19.23个百分点，说明局部序列上下文对二级结构预测非常重要。E类召回率最低，可能同时受到类别较少和β折叠上下文更复杂的影响。最终只保留 `checkpoints/cnn_best.pt`。

结果文件包括：

- `figures/mlp_loss.png`、`figures/cnn_loss.png`
- `figures/mlp_confusion_matrix.png`、`figures/cnn_confusion_matrix.png`
- `figures/model_comparison.png`
- `results/data_summary.json`、`results/*_metrics.json`、`results/model_comparison.json`

## 局限性

- 数据按随机蛋白质划分，未做序列同源性去冗余；高度相似但不同ID的蛋白仍可能跨集合。
- MLP不建模上下文；CNN仅覆盖有限局部窗口，不能表达长距离相互作用。
- 仅使用一级序列，没有进化谱、模板或预训练蛋白语言模型特征。
- 本实验优先保证CPU可运行和课程流程完整，没有进行大规模超参数搜索。
- 当前虚拟环境约737 MB，主要来自课程明确要求的PyTorch；代码、图表、结果和checkpoint本身不足1 MB。
