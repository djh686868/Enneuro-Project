# EnNeuro Web 端可视化 — 架构分析文档

> 分析日期：2026-06-11  
> 文档版本：v1.0  
> 覆盖范围：现有框架能力评估、Web 可视化可行性分析、关键接口与差距识别

---

## 一、现有框架概览

EnNeuro 是一个基于纯 Python/NumPy 的轻量级 CNN 框架，可选 CuPy 加速。框架由 9 个模块组成，已具备完整的训练、推理、可视化、模型解释能力和初步的 HTTP 服务层。

```
code/eneuro/
├── base/           # 自动微分引擎（Tensor、Function、Parameter）
├── nn/             # 网络层、模型容器、损失函数、优化器
├── data/           # 数据集抽象与 DataLoader（含异步）
├── train/          # Trainer 训练循环、Visualizer 可视化
├── utils/          # 钩子系统、序列化、静态可视化
├── explainability/ # GradCAM、Guided Backprop
├── ao/             # 算子融合（自动图优化）
└── global_config.py
code/serving/       # FastAPI HTTP 服务层（当前为推理专用）
```

---

## 二、各模块能力与 Web 可视化相关性

### 2.1 模型架构设计（`nn/module.py`）

**现有能力：**
- `Sequential`、`MLP`、`CNNWithPooling`、`ResidualBlock` 等预置模型
- `Layer.to_dict()` / `Layer.from_dict()` 实现完整模型序列化
- `Layer.__setattr__` 钩子自动收集参数，支持运行时反射层信息

**Web 可视化相关性：高**

模型字典结构可直接转换为 JSON 在前端渲染。`Layer` 的元数据（层类型、参数形状、超参数）已在 `to_dict()` 中保存，是前端拖拽建模的数据基础。

**差距：**
- 没有统一的"模型配置 Schema"——反序列化依赖硬编码的类名映射
- 无法从纯 JSON 配置自动实例化自定义层组合（需构建注册表）

---

### 2.2 数据集加载（`data/`）

**现有能力：**
- `Dataset` 抽象类 + 自定义 `prepare()` / `__getitem__`
- `DataLoader`（同步）+ `AsyncDataLoader`（多进程预取）
- 支持 shuffle、drop_last、batch 聚合

**Web 可视化相关性：高**

DataLoader 的批次迭代逻辑可包装为异步生成器，通过 Server-Sent Events (SSE) 向前端推送进度。但当前 DataLoader 是纯阻塞同步设计，不可直接挂入 async FastAPI handler。

**差距：**
- 无内置数据集（MNIST 等需手动下载）
- 无文件上传接口（用户无法通过 Web 上传数据集）
- `AsyncDataLoader` 使用 `multiprocessing`，在 FastAPI async context 中需要额外隔离

---

### 2.3 训练循环（`train/trainer.py`）

**现有能力：**
- `Trainer.train()` 封装完整训练循环（前向、损失、反向、更新、验证）
- 早停机制（patience、mode、权重恢复）
- `Visualizer` 收集每 epoch 的 loss/acc/时间数据

**Web 可视化相关性：极高（核心改造点）**

**差距（最关键）：**
- `Trainer.train()` 是**同步阻塞**调用，不可在 HTTP handler 中直接调用
- 训练数据（loss、acc、epoch）只在内存中积累，无推送机制
- 无法中断训练（无 stop signal 接口）
- `Visualizer.plot_all()` 依赖 matplotlib，生成静态图片文件，无法流式推送到前端

---

### 2.4 钩子与特征图（`utils/hooks.py`）

**现有能力：**
- `capture_features(layer)` — 前向钩子捕获激活图
- `capture_backward(layer)` — 反向钩子捕获梯度
- `HookRegistry` — 全局层调用序列记录，支持 GradCAM 的后继层查找

**Web 可视化相关性：高**

钩子系统是全通道特征图实时展示的基础。激活数据为 NumPy 数组，可序列化为 base64 图像发送至前端。

**差距：**
- 钩子数据存储在 Python 对象上，没有"快照"机制
- 无法指定"每 N 步抓取一次特征图"的调度接口

---

### 2.5 GradCAM（`explainability/gradcam.py`）

**现有能力：**
- `GradCAM(model, target_layer).generate(input_tensor, class_idx)` 返回归一化热力图 `(H, W) ∈ [0,1]`
- 已修复 BatchNorm 后梯度均值为零的问题（使用后继层梯度）
- 输出为 NumPy 数组，可 `cv2.applyColorMap` 转为 BGR 图像

**Web 可视化相关性：高**

GradCAM 热力图结果是标准 NumPy 数组，只需编码为 PNG base64 即可在前端展示。

**差距：**
- 当前无 HTTP 接口暴露 GradCAM 推理能力
- 目标层只能在代码中指定，需要前端可配置

---

### 2.6 现有 HTTP 服务层（`serving/`）

**现有能力：**
```
GET  /ping         → {"status": "success"}
GET  /health       → 状态 + 模型版本 + 运行时间
POST /predict      → 批量推理（List[List[float]]）
```
- 基于 FastAPI + Uvicorn
- Pydantic Schema 验证

**Web 可视化相关性：中（需大幅扩展）**

现有服务层仅覆盖推理场景，无训练、数据管理、可视化流式推送的任何接口。Predictor 中使用的是占位符虚拟模型，未集成真实框架模型。

**差距清单：**
- 无模型定义/配置接口
- 无数据集上传/选择接口
- 无训练启动/停止/状态接口
- 无 SSE / WebSocket 训练进度推送
- 无特征图/GradCAM 按需生成接口

---

## 三、差距汇总矩阵

| Web 功能需求 | 现有支撑 | 差距等级 | 主要工作 |
|---|---|---|---|
| 模型架构设计（前端拖拽） | `to_dict` / `from_dict` | **高** | 层注册表 + JSON Schema + 前端 |
| 预置模型引用 | Sequential/MLP/CNN | **中** | 模型目录 API |
| 数据集上传与加载 | Dataset 抽象类 | **高** | 文件上传、路径管理、前端 UI |
| 异步训练启动 | Trainer（同步） | **极高** | 线程/进程隔离 + 任务管理器 |
| 训练曲线实时推送 | Visualizer（静态） | **极高** | SSE/WebSocket + 前端图表 |
| 训练中途停止 | 无 | **高** | 停止信号 + Trainer 改造 |
| GradCAM 可视化 | GradCAM（离线） | **中** | HTTP 接口 + base64 编码 |
| 全通道特征图 | hooks（离线） | **中** | HTTP 接口 + 快照调度 |
| 模型保存/加载 | StateDict | **低** | 文件管理 API |

---

## 四、技术选型建议

### 4.1 后端

| 组件 | 建议方案 | 原因 |
|---|---|---|
| Web 框架 | **FastAPI**（已有） | 原生异步、Pydantic 验证、OpenAPI 文档自动生成 |
| 训练异步化 | `concurrent.futures.ProcessPoolExecutor` | 训练是 CPU/GPU 密集型，需独立进程 |
| 实时推送 | **Server-Sent Events (SSE)**（首选）或 WebSocket | SSE 实现简单，单向推送足够训练曲线需求 |
| 任务管理 | 内存中 `TaskManager` dict | 项目规模不需要 Celery/Redis |
| 图像编码 | `base64` + `cv2` / `PIL` | GradCAM/特征图 → PNG → base64 字符串 |

### 4.2 前端

| 组件 | 建议方案 | 原因 |
|---|---|---|
| UI 框架 | **Vue 3 + Vite**（或 React） | 组件化，响应式，生态完善 |
| 图表库 | **ECharts**（首选）或 Chart.js | ECharts 支持实时数据流、大数据量、交互丰富 |
| 模型设计器 | 基于 **Vue Flow**（dagre 布局） | 开源拖拽节点图，适合神经网络 DAG 可视化 |
| HTTP 客户端 | `fetch` API + EventSource | SSE 原生 EventSource，推理用 fetch |
| 图像展示 | `<img src="data:image/png;base64,...">` | 直接渲染 base64 编码图像 |

### 4.3 部署结构

```
┌─────────────────────────────────────────────────┐
│                   浏览器 (前端)                   │
│  Vue 3 SPA                                        │
│  ├── 模型设计器（Vue Flow）                        │
│  ├── 训练控制台（ECharts 实时曲线）                │
│  └── 可视化面板（GradCAM / 特征图）               │
└────────────────────┬────────────────────────────┘
                     │ HTTP / SSE
┌────────────────────▼────────────────────────────┐
│            EnNeuro Web API Server                │
│  FastAPI + Uvicorn                                │
│  ├── /api/models    — 模型管理                   │
│  ├── /api/datasets  — 数据集管理                  │
│  ├── /api/training  — 训练任务（SSE 流）          │
│  └── /api/explain   — GradCAM / 特征图           │
└────────────────────┬────────────────────────────┘
                     │ 函数调用（同进程或子进程）
┌────────────────────▼────────────────────────────┐
│              EnNeuro 框架核心                     │
│  eneuro.nn / eneuro.train / eneuro.explainability │
└─────────────────────────────────────────────────┘
```

---

## 五、关键风险与缓解

| 风险 | 描述 | 缓解策略 |
|---|---|---|
| 训练阻塞主线程 | `Trainer.train()` 同步阻塞，FastAPI 无法响应其他请求 | 用 `ProcessPoolExecutor` 在独立进程运行训练 |
| 跨进程数据共享 | 子进程中的 Trainer 无法直接修改主进程内存 | 使用 `multiprocessing.Queue` 或共享文件传递指标 |
| GradCAM 内存占用 | 大模型的激活图可能占用大量内存 | 推理后立即释放钩子；图像降采样后再传输 |
| 模型反序列化安全 | `from_dict` 依赖类名映射，可能存在注入风险 | 建立白名单层注册表，拒绝未知类名 |
| 前端状态同步 | 多用户同时发起训练导致状态混乱 | 每个训练任务分配唯一 `task_id`，SSE 流绑定 task_id |

---

*文档结束*
