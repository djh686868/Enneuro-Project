# EnNeuro AutoML Agent 使用指南

所有启动脚本都接受 `--strategy random|local|evolutionary|beyesian`，默认使用 `random`。策略决定下一次试验的超参数，启动脚本决定数据集、模型和训练预算。Python 调用方式与策略细节见 [Agent 接口](automl_agent_api.md)。

## 环境与快速验证

在包含 `demos/`、`code/` 和 `requirements.txt` 的 AutoML 工作树根目录运行。以下命令使用 PowerShell 和当前已激活的 Python 环境。

```powershell
python -m pip install -r requirements.txt
python -m pip install matplotlib
$env:PYTHONPATH = 'code'
```

可以先用内置的 MNIST 形状合成数据检查安装。这个命令的结果只用于检查运行流程。

```powershell
python demos/automl_mnist_agent.py --strategy random --trials 2 --epochs 1
```

CUDA 入口需要 NVIDIA GPU、与环境匹配的 CuPy 和 Matplotlib。若使用本项目的 CUDA 12.6 环境，在当前 PowerShell 设置工具包路径，再安装可选依赖。

```powershell
$env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6'
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
python -m pip install -r requirements-cuda.txt
```

## MNIST 分类

CPU 与 CUDA 入口可共用项目原有的 `mnist.pkl`。从 AutoML 工作树使用主工作区现成数据时，可以先检查文件内容。只读取信任来源的 pickle。
`prepare_automl_data.py` 读取已有本地数据。原始 MNIST 文件与车辆数据需要提前准备。

```powershell
$mnist = '../../code/tests/testdata/MNIST_data/mnist.pkl'
python demos/prepare_automl_data.py mnist --data $mnist
```

如果原始数据是 IDX 文件，将训练和测试各自的图像、标签共四个文件放入同一目录。文件名分别为 `train-images-idx3-ubyte`、`train-labels-idx1-ubyte`、`t10k-images-idx3-ubyte`、`t10k-labels-idx1-ubyte`，也可带 `.gz` 后缀。转换需要指定新文件作为输出。转换后把生成的路径传给 `--data`。

```powershell
$mnist = 'data/mnist.pkl'
python demos/prepare_automl_data.py mnist-idx --source-dir 'D:/datasets/mnist-idx' --output $mnist
```

CPU 入口使用内层验证错误率选参。下面的脚本用相同的样本数量和种子分别启动四种策略，结果写入不同目录。将它保存为工作树根目录下的 `run_mnist_strategies.ps1`，运行时通过 `-Data` 指定数据文件。

```powershell
param([Parameter(Mandatory=$true)][string]$Data)

foreach ($strategy in 'random', 'local', 'evolutionary', 'beyesian') {
    python demos/automl_mnist_agent.py --data $Data --strategy $strategy `
      --trials 12 --seed 42 --split-seed 0 --train-samples 256 --val-samples 64 `
      --test-samples 256 --epochs 1 --output "artifacts/mnist_cpu_${strategy}"
    if ($LASTEXITCODE -ne 0) { throw "MNIST Agent failed for $strategy" }
}
```

```powershell
.\run_mnist_strategies.ps1 -Data $mnist
```

CUDA 入口训练 EnNeuro LeNet。以下示例限制样本数来检查启动流程。删去三个样本数选项时，入口默认使用 55,000 个训练样本、5,000 个验证样本和 10,000 个测试样本。

```powershell
python -u demos/automl_mnist_cuda.py --data $mnist --strategy beyesian --backend cupy `
  --trials 12 --budget-minutes 20 --epochs 1 --train-samples 256 `
  --val-samples 128 --test-samples 128 --output artifacts/mnist_cuda_beyesian
```

## 车辆行为克隆

数据目录直接包含 `driving_log.csv` 和 `IMG/`。CSV 用 `center`、`left`、`right` 列引用三路摄像头图像。图片路径应相对数据目录，例如 `IMG/center_001.jpg`。CPU 转向示例需要 `steering` 标签。CUDA 四目标回归还需要 `throttle`、`brake` 和 `speed`。

如果项目的车辆 ZIP 内已经包含 `data/`，解压目标是工作树根目录。将示例中的 ZIP 路径替换为实际路径。已经解压的数据可直接指定给 `$car`。车辆和 DonkeyCar 的检查命令会读取标签、检查图像路径并汇总数据划分，图像像素在实际训练时读取。

```powershell
Expand-Archive -LiteralPath 'D:/datasets/car_behavioral_cloning_data.zip' -DestinationPath .
$car = 'data'
python demos/prepare_automl_data.py car --data-dir $car
python demos/prepare_automl_data.py car --data-dir $car --telemetry
```

CPU 入口使用验证转向 MSE 选参。

```powershell
python demos/automl_car_agent.py --data-dir $car --device cpu --strategy local `
  --trials 8 --epochs 1 --image-width 32 --image-height 16 --max-samples 120 `
  --output artifacts/car_cpu_local
```

CUDA 入口用 ResNet18 同时预测四项驾驶标签，以标准化验证 MSE 选参。

```powershell
python -u demos/automl_car_cuda.py --data-dir $car --strategy evolutionary `
  --trials 12 --budget-minutes 120 --epochs 1 --refit-epochs 1 `
  --output artifacts/car_cuda_evolutionary
```

## 历史 DonkeyCar 转向数据

将历史数据的 JPEG 放在同一个目录中，文件名用 `数字ID_转向角.jpg`，例如 `42_-0.15.jpg`。至少需要三个不同的有效 ID。训练、验证和测试按 ID 顺序划分，相同 ID 的图像进入同一分区。启动脚本训练单输出 ResNet18，并用验证转向 MSE 选参。

```powershell
$donkey = '../../tests/test_donkey/data'
python demos/prepare_automl_data.py donkeycar --data-dir $donkey
python -u demos/automl_donkeycar_cuda.py --data-dir $donkey --strategy local `
  --trials 8 --budget-minutes 120 --epochs 1 --refit-epochs 1 `
  --output artifacts/donkeycar_cuda_local
```

要把车辆或 DonkeyCar 命令保存成批量启动脚本，可复用 MNIST 示例的循环，将循环内的命令换成相应入口。用 `--data-dir $Data` 接收目录参数，将 `$strategy` 传给 `--strategy`，并让输出目录包含策略名。接入其他数据格式时，可参照 [执行器接口](automl_agent_api.md#接入新数据集或模型) 将数据加载、训练和验证封装成一次试验。

## 预算与结果

- 每个命令只运行所选策略。比较策略时使用同一个入口，固定数据、种子、试验数和训练轮数。CPU 多次评估可加 `--runs`，每次使用连续的训练种子，并和固定超参数基线配对。
- 进化搜索的前 6 次尝试用于初始化种群，下一代至少需要 2 个成功父代。BO 需要先完成 5 个成功试验。CPU 的默认试验数只够初始化，选择这两种策略时应显式提高 `--trials`。CUDA 还需预留足够的搜索时间。
- CUDA 的 `--budget-minutes` 只计搜索，`--trials` 可额外限制尝试次数。到期前没有完成验证的试验不参与排名。最终重训和外层测试单独计时。CUDA 的 `--output` 必须是新目录，或者省略该选项使用带时间戳的默认路径。
- `report.json` 中的 `strategy`、`best_config` 和 `best_result` 分别记录所选策略、最优配置和验证结果。`events.jsonl` 保存逐步事件。CPU 入口的 `evaluation.json` 还记录各运行种子的外层测试结果，按 `random_search`、`local_search`、`evolutionary_search` 或 `beyesian_search` 区分策略，并用 `fixed_baseline` 标记固定基线。
- `anytime.svg` 显示截至每个预算点的最佳验证目标，横轴是 CPU 的尝试次数或 CUDA 的搜索用时。CUDA 最终重训成功后还会保存 `best_checkpoint.bin`。

原有实验配置与评估细节见 [基础 AutoML 调参实验说明](automl_basic_agent.md)。
