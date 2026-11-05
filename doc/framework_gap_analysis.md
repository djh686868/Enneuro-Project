# EnNeuro 框架现状与可扩展/优化内容分析

> 文档日期：2026-06-11  
> 对照依据：本学期进阶式挑战性综合项目 II 任务书

**难度说明**：★☆☆☆☆ 极易 / ★★☆☆☆ 较易 / ★★★☆☆ 中等 / ★★★★☆ 较难 / ★★★★★ 高难

---

## 一、框架现状速览

| 模块 | 路径 | 已实现核心内容 |
|------|------|---------------|
| 计算图 | `eneuro/ao/` | 图构建(Tracer)、拓扑排序、图执行器(GraphExecutor)、算子融合(PatternMatcher)、AutoCast/GradScaler |
| 基础算子 | `eneuro/base/functions.py` | Im2col+GEMM、Winograd 3×3 s=1 卷积、转置卷积、池化、BN、融合算子(FusedConvReLU/BNReLU)、GroupedConv/DepthwiseConv |
| 神经网络层 | `eneuro/nn/module.py` | Linear、Conv2d、Deconv2d、BatchNorm、Sequential、MLP、CNNWithPooling |
| 损失函数 | `eneuro/nn/loss.py` | CrossEntropy、MSE、SigmoidBCE |
| 优化器 | `eneuro/nn/optim.py` | SGD(+Momentum)、Adam，含 L1/L2 正则 |
| 训练器 | `eneuro/train/trainer.py` | Trainer、Evaluator、Early Stopping、AverageMeter/TimeMeter |
| 数据管道 | `eneuro/data/` | Dataset 基类、DataLoader、AsyncDataLoader（多进程预取） |
| 序列化 | `eneuro/utils/serializer.py` | JSON+NPZ 方式保存/加载模型和优化器，checkpoint 支持 |
| 可视化 | `eneuro/utils/visualization.py` | 训练曲线（loss/acc）、混淆矩阵 |
| 可解释性 | `eneuro/explainability/` | Grad-CAM、Guided Backpropagation、GuidedGradCAM |
| 服务化 | `code/serving/` | FastAPI HTTP 推理服务（含健康检查、benchmark client） |
| 钩子系统 | `eneuro/utils/hooks.py` | HookManager、capture_features/capture_gradients |
| 模型 | `code/tests/` | LeNet、ResNet-18、简单 CNN，已有 Donkeycar 场景测试 |

---

## 二、对照任务书的 Gap 分析

### 2.1 基本任务（必须完成，影响"中等"以上评级）

#### A. 数据处理模块 —— **缺失**

| 属性 | 内容 |
|------|------|
| 难度 | ★★☆☆☆ |
| 预期工时 | 1.5 ~ 2 天 |
| 关键风险 | MixUp/CutMix 需修改标签为软标签，Trainer 的 loss 计算需同步适配 |

任务书要求：随机裁剪、旋转、颜色抖动、MixUp、CutMix。  
现状：`Dataset` 仅持有 `transform` 回调接口，**无任何内置增强算子**，用户需手写 lambda。  
需补充：
- `eneuro/data/transforms.py`：`RandomCrop`、`RandomHorizontalFlip`、`RandomRotation`、`ColorJitter`、`Normalize`、`Compose`
- `eneuro/data/transforms.py`：`MixUp`（在 DataLoader/Trainer 层 collate 阶段实施）
- `eneuro/data/transforms.py`：`CutMix`（同上，需修改标签为 one-hot soft label）

**参考文献：**
- MixUp 原文：Zhang et al., *mixup: Beyond Empirical Risk Minimization*, ICLR 2018. https://arxiv.org/abs/1710.09412
- CutMix 原文：Yun et al., *CutMix: Training Strategy that Makes Use of Sample Mixing*, ICCV 2019. https://arxiv.org/abs/1905.04899
- 数据增强综述：Shorten & Khoshgoftaar, *A survey on Image Data Augmentation for Deep Learning*, Journal of Big Data, 2019. https://doi.org/10.1186/s40537-019-0197-0
- torchvision transforms 参考实现（接口设计对标）：https://pytorch.org/vision/stable/transforms.html

---

#### B. 模型量化 —— **缺失**

| 属性 | 内容 |
|------|------|
| 难度 | ★★★★☆ |
| 预期工时 | 3 ~ 4 天（PTQ）；QAT 另需 3 天 |
| 关键风险 | 激活范围校准需选取代表性数据；INT8 矩阵乘在纯 NumPy 下无法获得实际加速，需结合 C 扩展或 numba 才能体现性能收益 |

任务书要求：模型量化（浮点 → INT8）。  
现状：`ao/cast.py` 中有 `AutoCastManager` 支持 float16 AMP，但**无 INT8 量化路径**。  
需补充：
- 对称/非对称线性量化：`scale = max(|W|) / 127`
- 逐层 Post-Training Quantization（PTQ）：校准数据集计算激活范围
- 量化感知训练（QAT，可选进阶）：伪量化节点插入计算图

**参考文献：**
- 量化综述（必读）：Gholami et al., *A Survey of Quantization Methods for Efficient Neural Network Inference*, 2021. https://arxiv.org/abs/2103.13630
- PTQ 经典方法：Nagel et al., *Data-Free Quantization Through Weight Equalization and Bias Correction*, ICCV 2019. https://arxiv.org/abs/1906.04721
- QAT 原始论文：Jacob et al., *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*, CVPR 2018. https://arxiv.org/abs/1712.05877
- TensorRT INT8 量化白皮书（工程参考）：NVIDIA, *TensorRT Best Practices Guide — INT8 Calibration*. https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#int8-calibration

---

#### C. 服务化部署（gRPC）—— **部分实现**

| 属性 | 内容 |
|------|------|
| 难度 | ★★★☆☆ |
| 预期工时 | 1.5 ~ 2 天 |
| 关键风险 | proto 文件版本与 `grpcio-tools` 的兼容性；图片数据序列化（bytes vs repeated float）的选择影响延迟 |

现状：已有 FastAPI HTTP 服务和 TCP 裸 socket 服务（`tcp_server.py`），**无 gRPC**。  
需补充：
- 编写 `proto/eneuro.proto`，定义 `PredictRequest`/`PredictResponse`
- `serving/grpc_server.py`：基于 `grpcio` 实现同名接口
- 对比 HTTP vs gRPC 延迟/吞吐量

**参考文献：**
- gRPC 官方文档（Python 快速入门）：https://grpc.io/docs/languages/python/quickstart/
- Protocol Buffers 语言指南：https://protobuf.dev/programming-guides/proto3/
- HTTP/REST vs gRPC 性能对比：Brito & Cacho, *REST vs gRPC: An Empirical Comparison of Performance*, CASCON 2019. https://dl.acm.org/doi/10.5555/3400745.3400754
- MLServer（生产级推理框架参考，含 gRPC 实现）：https://mlserver.readthedocs.io/en/latest/

---

#### D. 指标评价模块 —— **不完整**

| 属性 | 内容 |
|------|------|
| 难度 | ★★☆☆☆ |
| 预期工时 | 0.5 ~ 1 天 |
| 关键风险 | 多分类 F1 的 macro/micro/weighted 语义容易混淆，建议对照 sklearn 输出验证 |

现状：`Evaluator` 仅计算 Top-1 准确率。  
需补充：
- `eneuro/train/metrics.py`：Precision、Recall、F1（macro/micro/weighted）、Top-5 Accuracy
- 多标签场景下的 mAP（可选，用于目标检测拓展）
- 与 sklearn 结果对比验证

**参考文献：**
- 评价指标综述：Sokolova & Lapalme, *A systematic analysis of performance measures for classification tasks*, Information Processing & Management, 2009. https://doi.org/10.1016/j.ipm.2009.03.002
- Top-K Accuracy 定义：Russakovsky et al., *ImageNet Large Scale Visual Recognition Challenge*, IJCV 2015. https://arxiv.org/abs/1409.0575
- mAP 计算方法（PASCAL VOC）：Everingham et al., *The Pascal Visual Object Classes Challenge: A Retrospective*, IJCV 2015. https://link.springer.com/article/10.1007/s11263-014-0733-5
- scikit-learn 指标 API 文档（对比验证参考）：https://scikit-learn.org/stable/modules/model_evaluation.html

---

#### E. 可视化模块完善 —— **部分实现**

| 属性 | 内容 |
|------|------|
| 难度 | ★★☆☆☆（离线图表）/ ★★★★☆（Web 实时） |
| 预期工时 | 离线增强 0.5 天；Web 实时界面 2 ~ 3 天 |
| 关键风险 | WebSocket 推送频率过高会阻塞训练主循环，需在独立线程/进程中推送 |

现状：`Visualizer` 已有训练曲线和混淆矩阵，`Grad-CAM` 已实现。  
缺少：
- **权重分布直方图**（每 epoch 或每 N step）
- **梯度变化曲线/梯度 norm 监控**（已有 hooks，需接入 Visualizer）
- **Web 端实时展示**（当前全为离线 matplotlib）：建议基于 WebSocket + 轻量前端（可用 Streamlit 或手写 HTML+Chart.js）实时推送训练指标

**参考文献：**
- TensorBoard 设计论文（可视化训练指标的工业实践参考）：Wongsuphasawat et al., *Visualizing Dataflow Graphs of Deep Learning Models in TensorFlow*, IEEE TVCG 2018. https://arxiv.org/abs/1707.07356
- 梯度消失/爆炸可视化动机：Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*, AISTATS 2010. https://proceedings.mlr.press/v9/glorot10a.html
- Streamlit 文档（轻量 Web 可视化方案）：https://docs.streamlit.io/
- Chart.js（前端图表库，适合手写 HTML 方案）：https://www.chartjs.org/docs/latest/

---

### 2.2 可选拓展（影响"良好/优秀"评级）

#### 拓展 ① CNN 卷积效率优化 —— **部分完成，可继续深化**

| 属性 | 内容 |
|------|------|
| 难度 | ★★★★☆（Loop Tiling）/ ★★★★★（Winograd F(4×4)） |
| 预期工时 | Loop Tiling 1.5 天 + 对比实验 1 天；Winograd F(4×4) 3 ~ 4 天 |
| 关键风险 | NumPy 环境下 Loop Tiling 收益有限（GIL、内存分配开销），需用 numba JIT 或 Cython 才能体现；Winograd 扩展需推导新的变换矩阵 B/G/A |

现状：已实现 Im2col+GEMM、Winograd F(2×2, 3×3)、FusedConvReLU/BNReLU。  
可继续：
- **Loop Tiling（循环分块）**：在 `im2col_array` 或 GEMM 的 NumPy 路径中引入分块，提升 L1/L2 Cache 命中率
- **Winograd F(4×4, 3×3)**：将 tile 从 2×2 扩展到 4×4，乘法次数从 F(2×2) 的 16 次减少为 9 次（每 tile）
- **Depthwise 卷积优化**：当前 `depthwise_conv2d` 使用 GroupedConv 实现，可改为专用 channel-wise 循环

**参考文献：**
- Im2col+GEMM 原始方法：Chellapilla et al., *High Performance Convolutional Neural Networks for Document Processing*, IWNIST 2006. （工程出处，可通过 Google Scholar 检索）
- Winograd 卷积原文：Lavin & Gray, *Fast Algorithms for Convolutional Neural Networks*, CVPR 2016. https://arxiv.org/abs/1509.09308
- Loop Tiling 与 Cache 优化：Lam et al., *The Cache Performance and Optimizations of Blocked Algorithms*, ASPLOS 1991. https://dl.acm.org/doi/10.1145/106972.106981
- GEMM 优化综述（含分块策略）：Goto & van de Geijn, *Anatomy of High-Performance Matrix Multiplication*, ACM TOMS 2008. https://dl.acm.org/doi/10.1145/1356052.1356053
- numba JIT 文档（Python 层加速参考）：https://numba.readthedocs.io/en/stable/user/5minguide.html

---

#### 拓展 ② 模型算法优化（BatchNorm/ResNet 已有，可完善权重初始化与正则化）

| 属性 | 内容 |
|------|------|
| 难度 | ★★☆☆☆（权重初始化/Dropout）/ ★★★☆☆（ResNet 提升为内置层） |
| 预期工时 | 权重初始化 0.5 天；Dropout 0.5 天；ResNet 整合 1 天 |
| 关键风险 | Kaiming 初始化针对 ReLU 激活，与 sigmoid/tanh 初始化策略不同，需在 `Conv2d.__init__` 中按 `activation` 参数自动选择 |

现状：BatchNorm、ResidualBlock、ResNet-18 已在测试文件中实现。  
缺少：
- **系统性权重初始化**：当前初始化散落在 `module.py`，需统一提供 `kaiming_uniform_`、`xavier_uniform_`、`orthogonal_` 等接口至 `eneuro/nn/init.py`
- **Dropout 层**：ResNet 训练的正则化工具，当前未实现
- 将 ResidualBlock/ResNet-18 从测试文件提升为**框架内置层**

**参考文献：**
- Xavier 初始化：Glorot & Bengio, *Understanding the difficulty of training deep feedforward neural networks*, AISTATS 2010. https://proceedings.mlr.press/v9/glorot10a.html
- Kaiming 初始化：He et al., *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*, ICCV 2015. https://arxiv.org/abs/1502.01852
- Dropout：Srivastava et al., *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*, JMLR 2014. https://jmlr.org/papers/v15/srivastava14a.html
- Batch Normalization 原文：Ioffe & Szegedy, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*, ICML 2015. https://arxiv.org/abs/1502.03167
- ResNet 原文：He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016. https://arxiv.org/abs/1512.03385

---

#### 拓展 ③ 计算图优化（已有算子融合，可扩展）

| 属性 | 内容 |
|------|------|
| 难度 | ★★★☆☆（常量折叠/CSE）/ ★★★★★（内存复用分析/图调度） |
| 预期工时 | 常量折叠+CSE 2 天；完整内存规划 4 ~ 5 天 |
| 关键风险 | 内存生命周期分析需要精确的拓扑依赖信息，与动态图的 inplace 操作存在冲突 |

现状：`ao/pattern.py` 实现了基于模式匹配的算子融合（Conv-BN-ReLU 等）。  
可继续：
- **常量折叠**：识别计算图中仅由常量参数组成的子图，在编译期计算并替换
- **公共子表达式消除（CSE）**：检测重复的 Function 节点，共享输出
- **内存复用分析**：根据生命周期分析分配临时 buffer 池，减少 peak memory

**参考文献：**
- TVM 计算图优化（算子融合/常量折叠的工业实现）：Chen et al., *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*, OSDI 2018. https://arxiv.org/abs/1802.04799
- XLA 编译器（Google 的计算图优化方案）：Leary & Wang, *XLA: TensorFlow, Compiled*, TensorFlow Dev Summit 2017. https://www.tensorflow.org/xla
- 内存优化综述：Chen et al., *Training Deep Nets with Sublinear Memory Cost*, 2016. https://arxiv.org/abs/1604.06174
- 算子融合原理：Ragan-Kelley et al., *Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines*, PLDI 2013. https://dl.acm.org/doi/10.1145/2491956.2462176

---

#### 拓展 ④ 数据管道并行化（已有 AsyncDataLoader，可完善）

| 属性 | 内容 |
|------|------|
| 难度 | ★★☆☆☆ |
| 预期工时 | 1 天（含压力测试脚本） |
| 关键风险 | Windows 平台 `multiprocessing` spawn 模式下需注意 `if __name__ == '__main__'` 保护；CUDA tensor 不能直接在子进程中传递 |

现状：`AsyncDataLoader` 已通过 `multiprocessing` 实现多进程预取。  
可继续完善：
- **数据增强在 Worker 进程中执行**（当前 transform 在主进程，应移到 `_worker_loop`）
- **预取队列深度**自适应调节（根据 GPU 利用率动态调整 `num_workers`）
- 压力测试：统计数据读取成为训练瓶颈的场景

**参考文献：**
- PyTorch DataLoader 设计（生产者消费者模型参考）：https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
- DALI（GPU 数据预处理流水线，工业级参考）：NVIDIA Data Loading Library. https://docs.nvidia.com/deeplearning/dali/user-guide/docs/
- 生产者-消费者模型原理：Tanenbaum, *Modern Operating Systems*, Chapter 2 (IPC). https://www.pearson.com/en-us/subject-catalog/p/modern-operating-systems/P200000003295

---

#### 拓展 ⑤ 模型序列化扩展（ONNX 导出）

| 属性 | 内容 |
|------|------|
| 难度 | ★★★★☆ |
| 预期工时 | 3 ~ 4 天（仅支持 CNN 基本算子） |
| 关键风险 | ONNX opset 版本选择（建议 opset=17）；Im2col 路径需映射为 ONNX `Conv` 节点而非展开的矩阵乘；动态 batch size 处理 |

现状：已有 JSON+NPZ 格式，`ao/tracer.py` 可追踪计算图。  
可继续：
- 利用 `Graph` 数据结构将节点映射为 ONNX `NodeProto`
- 关键算子映射：`Im2col+GEMM → onnx::Conv`，`BatchNorm → onnx::BatchNormalization`，`ReLU → onnx::Relu`
- 使用 `onnxruntime` 验证导出模型与原始模型的输出一致性

**参考文献：**
- ONNX 规范文档：https://onnx.ai/onnx/intro/concepts.html
- ONNX opset 17 算子列表：https://onnx.ai/onnx/operators/
- onnxruntime Python API：https://onnxruntime.ai/docs/get-started/with-python.html
- 模型互操作性综述：Vartak et al., *MISTIQUE: A System to Store and Query Model Intermediates for Model Diagnosis*, SIGMOD 2018. https://dl.acm.org/doi/10.1145/3183713.3196934

---

#### 拓展 ⑥ 容器化与服务编排

| 属性 | 内容 |
|------|------|
| 难度 | ★★★☆☆ |
| 预期工时 | Docker 封装 0.5 天；Kubernetes/KuFlow 集成 2 ~ 3 天 |
| 关键风险 | GPU 容器需要 `nvidia-container-toolkit`；Kubernetes 资源配额与 GPU 调度配置较复杂 |

可继续：
- Dockerfile 封装 `code/serving/` 的 HTTP/gRPC 服务
- `docker-compose.yml` 一键启动推理服务 + 监控
- 可选：Kubernetes `Deployment`/`Service` 配置，结合 KuFlow 任务调度

**参考文献：**
- Docker 官方文档（Python 应用容器化）：https://docs.docker.com/language/python/
- NVIDIA Container Toolkit（GPU 容器支持）：https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- Kubernetes 官方文档（Pod/Deployment 基础）：https://kubernetes.io/docs/concepts/workloads/
- Kubeflow（ML 工作流调度参考）：https://www.kubeflow.org/docs/started/introduction/

---

## 三、优先级与工时汇总

| # | 任务 | 评级影响 | 难度 | 预期工时 |
|---|------|---------|------|---------|
| 1 | 数据增强模块（transforms.py） | 中等+ | ★★☆☆☆ | 1.5 ~ 2 天 |
| 2 | 指标评价模块（metrics.py） | 中等+ | ★★☆☆☆ | 0.5 ~ 1 天 |
| 3 | INT8 量化（PTQ） | 中等+ | ★★★★☆ | 3 ~ 4 天 |
| 4 | gRPC 服务 | 中等+ | ★★★☆☆ | 1.5 ~ 2 天 |
| 5 | 权重分布/梯度 norm 可视化 | 良好+ | ★★☆☆☆ | 0.5 天 |
| 6 | Web 实时可视化 | 良好+ | ★★★★☆ | 2 ~ 3 天 |
| 7 | 权重初始化模块（nn/init.py） | 良好+ | ★★☆☆☆ | 0.5 天 |
| 8 | ResNet-18 提升为内置层 | 良好+ | ★★★☆☆ | 1 天 |
| 9 | Loop Tiling + 对比实验 | 优秀+ | ★★★★☆ | 2.5 天 |
| 10 | Winograd F(4×4,3×3) | 优秀+ | ★★★★★ | 3 ~ 4 天 |
| 11 | 计算图常量折叠/CSE | 优秀+ | ★★★☆☆ | 2 天 |
| 12 | ONNX 导出 | 优秀+ | ★★★★☆ | 3 ~ 4 天 |
| 13 | Docker 容器化 | 优秀+ | ★★★☆☆ | 0.5 ~ 1 天 |
| 14 | AsyncDataLoader 完善 | 良好+ | ★★☆☆☆ | 1 天 |

**总工时估算（按优秀目标全做）：约 23 ~ 29 人·天**  
建议团队按模块拆分并行，优先在前两周完成 #1~#5，后续穿插完成拓展项。

---

## 四、各模块扩展要点速查

### 数据增强 `eneuro/data/transforms.py`
```python
# 接口设计（仿 torchvision.transforms）
class Compose:
    def __init__(self, transforms): ...
    def __call__(self, img): ...   # img: np.ndarray (H,W,C)

class RandomCrop:
    def __init__(self, size, padding=0): ...

class RandomHorizontalFlip:
    def __init__(self, p=0.5): ...

class ColorJitter:
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0): ...

class MixUp:
    def __init__(self, alpha=0.2): ...
    def __call__(self, batch_x, batch_y): ...   # 在 collate 层调用，返回软标签

class CutMix:
    def __init__(self, alpha=1.0): ...
    def __call__(self, batch_x, batch_y): ...
```

### INT8 量化 `eneuro/ao/quantize.py`
```
伪代码：
1. CalibrationPass(model, calib_loader):
   - 前向传播，收集每层输出的 min/max（或使用直方图）
   - 计算 scale = max_abs / 127, zero_point = 0 (对称)
2. QuantizeModel(model, scales):
   - 将每个 Conv/Linear 的 weight 替换为 int8 数组
   - 在 forward 中先 dequantize → compute → quantize
3. 验证精度：对比 fp32 baseline vs int8 模型在验证集的 Top-1 Acc
```

### gRPC 服务 `code/serving/proto/eneuro.proto`
```protobuf
syntax = "proto3";
service EnNeuroPredict {
  rpc Predict (PredictRequest) returns (PredictResponse);
}
message PredictRequest {
  bytes image_data = 1;
  int32 width = 2;
  int32 height = 3;
}
message PredictResponse {
  repeated float logits = 1;
  int32 class_id = 2;
  float confidence = 3;
}
```

### 指标模块 `eneuro/train/metrics.py`
```
核心函数：
- accuracy(y_pred, y_true, topk=(1,5))
- precision_recall_f1(y_pred, y_true, average='macro')
- confusion_matrix(y_pred, y_true, num_classes)
- mAP(pred_boxes, gt_boxes, iou_threshold=0.5)  # 可选，用于 YOLO 拓展
```

---

## 五、与其他框架对比实验设计

对照任务书"同等条件下训练时间、资源占用、FLOPS 对比"要求，建议在完成 ResNet-18 / VGG 训练后统一采集：

| 指标 | EnNeuro | PyTorch | TensorFlow/PaddlePaddle |
|------|---------|---------|------------------------|
| 单 epoch 训练时间 (s) | - | - | - |
| GPU 峰值显存 (MB) | - | - | - |
| 理论 FLOPS（卷积层汇总） | - | - | - |
| Top-1 Acc（CIFAR-10/Donkeycar） | - | - | - |

FLOPS 计算公式（Conv2d）：
```
FLOPS = 2 × Cout × Cin × KH × KW × OH × OW
```
可在 `GraphExecutor` 的 `forward` 中统计每个 Conv 节点的 FLOPS 并累加。

**对比实验参考文献：**
- MLPerf 基准测试（工业标准对比方法论）：Mattson et al., *MLPerf Training Benchmark*, MLSys 2020. https://arxiv.org/abs/1910.01500
- 框架性能对比研究：Shi et al., *Benchmarking State-of-the-Art Deep Learning Software Tools*, 2016. https://arxiv.org/abs/1608.01249
- FLOPS 计算方法参考：Molchanov et al., *Pruning Convolutional Neural Networks for Resource Efficient Inference*, ICLR 2017. https://arxiv.org/abs/1611.06440

---

## 六、参考文献总索引

> 按主题分类，括号内为文档中对应的章节编号。

### 数据增强
1. Zhang et al., *mixup: Beyond Empirical Risk Minimization*, ICLR 2018. https://arxiv.org/abs/1710.09412 （§A）
2. Yun et al., *CutMix: Training Strategy that Makes Use of Sample Mixing*, ICCV 2019. https://arxiv.org/abs/1905.04899 （§A）
3. Shorten & Khoshgoftaar, *A survey on Image Data Augmentation for Deep Learning*, 2019. https://doi.org/10.1186/s40537-019-0197-0 （§A）

### 模型量化
4. Gholami et al., *A Survey of Quantization Methods*, 2021. https://arxiv.org/abs/2103.13630 （§B）
5. Nagel et al., *Data-Free Quantization*, ICCV 2019. https://arxiv.org/abs/1906.04721 （§B）
6. Jacob et al., *Quantization and Training of Neural Networks*, CVPR 2018. https://arxiv.org/abs/1712.05877 （§B）

### 卷积算法优化
7. Lavin & Gray, *Fast Algorithms for Convolutional Neural Networks*, CVPR 2016. https://arxiv.org/abs/1509.09308 （§拓展①）
8. Lam et al., *Cache Performance of Blocked Algorithms*, ASPLOS 1991. https://dl.acm.org/doi/10.1145/106972.106981 （§拓展①）
9. Goto & van de Geijn, *High-Performance Matrix Multiplication*, ACM TOMS 2008. https://dl.acm.org/doi/10.1145/1356052.1356053 （§拓展①）

### 模型结构与初始化
10. Glorot & Bengio, *Understanding the difficulty of training deep feedforward networks*, AISTATS 2010. https://proceedings.mlr.press/v9/glorot10a.html （§E, §拓展②）
11. He et al., *Delving Deep into Rectifiers*, ICCV 2015. https://arxiv.org/abs/1502.01852 （§拓展②）
12. He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016. https://arxiv.org/abs/1512.03385 （§拓展②）
13. Ioffe & Szegedy, *Batch Normalization*, ICML 2015. https://arxiv.org/abs/1502.03167 （§拓展②）
14. Srivastava et al., *Dropout*, JMLR 2014. https://jmlr.org/papers/v15/srivastava14a.html （§拓展②）

### 计算图与编译优化
15. Chen et al., *TVM: An Automated End-to-End Optimizing Compiler*, OSDI 2018. https://arxiv.org/abs/1802.04799 （§拓展③）
16. Chen et al., *Training Deep Nets with Sublinear Memory Cost*, 2016. https://arxiv.org/abs/1604.06174 （§拓展③）

### 评价指标
17. Russakovsky et al., *ImageNet Large Scale Visual Recognition Challenge*, IJCV 2015. https://arxiv.org/abs/1409.0575 （§D）
18. Sokolova & Lapalme, *Performance measures for classification tasks*, 2009. https://doi.org/10.1016/j.ipm.2009.03.002 （§D）

### 性能对比与基准测试
19. Mattson et al., *MLPerf Training Benchmark*, MLSys 2020. https://arxiv.org/abs/1910.01500 （§五）
20. Shi et al., *Benchmarking Deep Learning Software Tools*, 2016. https://arxiv.org/abs/1608.01249 （§五）

---

*文档将随开发进展持续更新。*
