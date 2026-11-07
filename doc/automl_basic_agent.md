# 基础 AutoML 调参 Agent

本分支实现了一个依赖很少的顺序式超参数优化（HPO）闭环，供 EnNeuro
学习和演示使用。它遵循 AutoML 课程中的基本流程：在低维配置空间采样，
执行一次训练，观察验证指标，再在固定预算内选择下一组配置。

核心 API 是 `SequentialRandomSearch`。它会记录 `observe`、`decide`、
`execute`、`feedback` 四类事件，并同时生成完整 JSON 报告和 JSONL 事件流。
默认搜索空间让学习率在 log10 空间均匀采样，同时把 batch size 和 optimizer
作为离散选项处理。

从仓库根目录运行真实 MNIST 小规模搜索：

```powershell
$env:PYTHONPATH = "code"
python demos/automl_mnist_agent.py `
  --data "D:\Undergraduate\WZH\EnNeuro\tmp\Enneuro-Project\code\tests\testdata\MNIST_data\mnist.pkl" `
  --trials 4 --train-samples 256 --test-samples 256 --epochs 1 `
  --output artifacts/automl_mnist_real
```

车辆行为克隆演示会直接读取 `car_behavioral_cloning_data.zip` 中的
`driving_log.csv` 和三路摄像头图像，并应用标准的左右转向修正；所有图像
不会一次性物化成一个大数组：

```powershell
$env:PYTHONPATH = "code"
python demos/automl_car_agent.py `
  --zip "D:\Undergraduate\WZH\EnNeuro\car_behavioral_cloning_data.zip" `
  --device cpu --trials 3 --epochs 1 --image-width 32 --image-height 16 `
  --output artifacts/automl_car_full_cpu
```

车辆任务的调参目标是验证集 MSE；报告还包含验证/测试 MAE、样本数量、
耗时和最终选择的配置。闭环稳定后，可以在同一 Executor 接口中替换为
更大的 CNN 或 CUDA 后端。

## 按文件理解代码

1. `protocol.py` 定义搜索空间、trial 结果和事件对象。
2. `search.py` 串起 observe、decide、execute、feedback 四步。
3. `logger.py` 将事件写成 JSON 和 JSONL。
4. `executors.py` 提供合成目标和轻量 MNIST 执行器。
5. `car_executor.py` 负责 ZIP/CSV、三路摄像头、EnNeuro 回归训练和指标统计。
6. 两个 `demos` 文件把上述组件组合成可复现命令。
7. `tests/test_agent_automl.py` 检查复现性、采样范围和真实 minibatch 执行。
