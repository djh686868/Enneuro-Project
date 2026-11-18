# 基础 AutoML 调参 Agent

本分支实现了一个依赖很少的顺序式超参数优化（HPO）闭环，供 EnNeuro
学习和演示使用。它遵循 AutoML 课程中的基本流程：在低维配置空间采样，
执行一次训练，观察验证指标，再在固定预算内选择下一组配置。

下文保留随机搜索基线的实验配置和评估背景。新增搜索策略的 Python 接口见
[Agent 接口](automl_agent_api.md)，数据准备和各入口启动命令见
[Agent 使用指南](automl_agent_guide.md)。

从仓库根目录运行真实 MNIST 小规模搜索：

```powershell
$env:PYTHONPATH = "code"
python demos/automl_mnist_agent.py `
  --data "D:\Undergraduate\WZH\EnNeuro\tmp\Enneuro-Project\code\tests\testdata\MNIST_data\mnist.pkl" `
  --trials 4 --train-samples 256 --val-samples 64 --test-samples 256 --epochs 1 `
  --inner-folds 5 --runs 8 --output artifacts/automl_ch2_mnist
```

### MNIST CUDA 调参

在 AutoML 工作区根目录用 PowerShell 启动。CUDA 版训练 EnNeuro LeNet，默认使用
55,000 张训练图像和 5,000 张验证图像。选定配置后用这 60,000 张开发数据重训，
再用官方 10,000 张测试图像评分。

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:CUPY_CACHE_DIR = Join-Path $PWD 'artifacts\automl_mnist_cuda_cupy_cache'
& 'C:\Users\Administrator\.conda\envs\EnNeuro\python.exe' -X utf8 -u demos/automl_mnist_cuda.py `
  --data '../../code/tests/testdata/MNIST_data/mnist.pkl' --budget-minutes 20 --backend cupy
```

终端实时显示配置、每个 epoch 的训练损失和验证准确率，以及剩余搜索时间。
截止时间按批次检查，尚未完成验证的试验不进入最优记录。最终重训与测试单独计时。
启动时打印本次输出目录，里面有实时事件日志 `events.jsonl`、汇总 `report.json`、
以搜索实际用时为横轴的 `anytime.svg`，以及完成重训后的 `best_checkpoint.bin`。

### 车辆 CPU 转向演示

车辆行为克隆演示读取解压后的 `driving_log.csv` 和 `IMG/` 下的三路摄像头
图像，并对左右摄像头应用转向修正。执行器按批次读取所需图像。

首次准备数据时，若项目根目录还没有 `data/`，在该目录运行以下 PowerShell
命令。ZIP 已自带 `data/` 这一层，解压目标是项目根目录。

```powershell
Expand-Archive -LiteralPath 'D:\Undergraduate\WZH\EnNeuro\car_behavioral_cloning_data.zip' -DestinationPath .
```

启动调参时，`--data-dir` 默认为项目根目录下的 `data/`。数据放在其他位置时，
将它设为同时包含 `driving_log.csv` 和 `IMG/` 的目录。

```powershell
$env:PYTHONPATH = "code"
python demos/automl_car_agent.py `
  --data-dir data `
  --device cpu --trials 3 --runs 3 --epochs 1 --image-width 32 --image-height 16 `
  --output artifacts/automl_ch2_car
```

### 车辆 CUDA 调参

限时版沿用项目 DonkeyCar 的 ResNet18 骨干，回归头依次预测转向、油门、
刹车和速度。默认使用三路摄像头和 96×48 输入，搜索学习率、批大小
（16 或 32）和优化器（Adam 或 SGD）。搜索按时间预算停止，每组训练
10 轮后验证一次。最佳配置重新初始化，在完整开发集上训练 30 轮。

24,108 张图像分为 19,284 张训练、2,412 张验证和 2,412 张外层测试。
同一条驾驶记录的三路图像总是一起划分。外层测试固定为 CSV 末段，
内层验证从开发集留出一段含刹车记录的连续路段。四个标签分别用内层
训练集的均值和标准差标准化，以四项标准化验证 MSE 的平均值选择配置。

RTX 4060 上一次全量搜索中，每组 3 轮的训练与验证用时约 290–469 秒。
据此估算，每组 10 轮约需 16–26 分钟。搜索默认有 120 分钟预算，
到时仍未完成验证的候选不参与排名。快速验证可用 `--trials` 另设组数上限。
最终重训与测试另计，30 轮预计约需 55–90 分钟。

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:CUPY_CACHE_DIR = Join-Path $PWD 'artifacts\automl_mnist_cuda_cupy_cache'
& 'C:\Users\Administrator\.conda\envs\EnNeuro\python.exe' -X utf8 -u demos/automl_car_cuda.py `
  --data-dir data --epochs 10 --refit-epochs 30 --budget-minutes 120
```

终端显示试验配置、每轮训练 MSE、候选验证 MSE 和剩余时间。搜索结束后，
以最佳配置在训练和验证数据的并集上重训，完成后对外层留出集评分一次。
启动时显示输出目录，其中包括墙钟时间阶梯图 `anytime.svg`、实时事件
日志 `events.jsonl`、汇总 `report.json` 和重训 checkpoint。报告中的
`outer_test.train_mse_by_epoch` 保存重训的逐轮训练 MSE，可以查看后期训练
损失是否仍在下降。报告还按原始标签单位记录四个目标的 MAE 和 RMSE。
Checkpoint 配置记录输入尺寸、输出顺序和反标准化所需的统计量。

### 原 DonkeyCar CUDA 调参

从 AutoML 工作树运行以下命令。原数据位于主工作区的
`tests/test_donkey/data`，包含 6,200 张 `数字ID_转向角.jpg` 图像。
脚本按数值 ID 做连续留出，比例约为 70% / 15% / 15%，重复 ID 始终
在同一集合。历史训练脚本使用随机划分，比较结果时应注明这一差异。

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:CUPY_CACHE_DIR = Join-Path $PWD 'artifacts\automl_mnist_cuda_cupy_cache'
& 'C:\Users\Administrator\.conda\envs\EnNeuro\python.exe' -X utf8 -u demos/automl_donkeycar_cuda.py `
  --data-dir '../../tests/test_donkey/data' --budget-minutes 120
```

默认使用 160×120 RGB 输入和单输出 ResNet18。每组训练 10 轮，用验证集
原始转向角 MSE 排名。搜索按 120 分钟截止，最佳配置用完整开发集重训
30 轮，再对外层测试集报告转向角 MSE 和 MAE。重训与测试单独计时。
运行目录中保存 `events.jsonl`、`report.json`、`anytime.svg` 和重训 checkpoint。

## 评估协议（AutoML MOOC 第 2 章）

- **内层模型评估。** MNIST 从原始训练分区中取开发样本，默认用固定留出集验证。
  `--inner-folds 5` 在相同开发数据上评估每个候选的五折平均错误率。
  车辆按原始 CSV 驾驶记录分组，同一行的各路图像进入同一集合。CPU 转向
  演示和 CUDA 四目标回归各用自己的留出协议和验证 MSE，见上文。
- **外层测试。** 两种方法锁定各自配置后，在训练和验证数据的并集上重新训练，
  然后各评一次外层测试集。MNIST 的测试数据来自原始测试分区。
  这是内层验证加单次外层留出，课件中的多外层折嵌套 CV 属于另一种协议。
- **Anytime 曲线。** 横轴是已尝试的配置数，纵轴是内层验证最优值。
  第一个有效值出现前留空，失败尝试仍计入预算。曲线采用右连续阶梯，
  多次运行逐点评估均值、样本标准差和标准误（SE），图中阴影为均值 ± SE。
  固定基线只评估一组参数，余下预算处沿用这一结果。
- **比较和统计。** 固定 Adam 配置与所选搜索策略共享数据划分、设备和每对的训练种子。
  每种方法的预算上限按配置尝试次数计，实际搜索耗时另记。
  外层测试主指标按种子配对，报告原始分数、平均和中位差值。所有配对有效且
  至少有六个非零配对差值时，报告精确双侧 Wilcoxon 符号秩检验。
  它依赖独立配对差值和零假设下的符号对称性，解释范围是固定数据拆分上的随机种子。
  MNIST 和车辆分开报告，两个任务的错误率和 MSE 采用不同尺度。

`report.json` 和 `events.jsonl` 保存首个种子的搜索记录。`evaluation.json`
保存所有种子的原始结果、外层测试指标、数据划分、预算、硬件环境和曲线数据。
安装 Matplotlib 后，演示同时输出 `anytime.svg`。

## 按文件理解代码

1. `protocol.py` 定义搜索空间、trial 结果和事件对象。
2. `search.py` 串起 observe、decide、execute、feedback 四步。
3. `logger.py` 将事件写成 JSON 和 JSONL。
4. `executors.py` 提供合成目标和轻量 MNIST 执行器。
5. `car_executor.py` 负责从数据目录读取 CSV 和图像、处理三路摄像头、
   执行 EnNeuro 回归训练并统计指标。
6. `evaluation.py` 负责外层评分、基线、多种子统计和阶梯曲线。
7. `demos/automl_*.py` 将数据集和训练组件组合成可复现命令。
8. `tests/test_agent_automl.py` 检查复现性、采样范围和 minibatch 执行。
