# EnNeuro Web 端可视化 — 实施计划文档

> 分析日期：2026-06-11  
> 文档版本：v1.0

---

## 一、目标与范围

### 目标

实现一个与 EnNeuro 框架深度集成的 Web 可视化平台，支持：

1. **模型架构设计/引用** — 在浏览器中拖拽构建或引用预置模型
2. **数据集加载** — 上传 / 选择数据集，预览样本
3. **模型训练** — 启动、监控、停止训练任务
4. **实时训练曲线** — 通过 SSE 流推送 loss/acc，前端 ECharts 动态更新
5. **推理可视化** — 按需生成 GradCAM 热力图 / 全通道特征图并展示

### 不在范围内（当前阶段）

- 分布式训练支持
- 用户认证/多租户
- 生产级容器化部署（Docker/K8s）

---

## 二、整体里程碑

| 阶段 | 编号 | 名称 | 主要交付物 | 预估工作量 |
|---|---|---|---|---|
| **Phase 1** | P1 | 后端基础服务 | 扩展 FastAPI，层注册表，任务管理器 | 3–4 天 |
| **Phase 2** | P2 | 训练异步化 + SSE | Trainer 改造，Queue 指标推送，SSE 端点 | 3–4 天 |
| **Phase 3** | P3 | 前端基础框架 | Vue 3 + Vite 项目，路由，布局 | 2–3 天 |
| **Phase 4** | P4 | 模型设计器 | Vue Flow 节点图，JSON 配置生成 | 3–4 天 |
| **Phase 5** | P5 | 数据集管理 | 文件上传，数据集预览 API + UI | 2–3 天 |
| **Phase 6** | P6 | 训练控制台 | 训练启停 UI + ECharts 实时曲线 | 2–3 天 |
| **Phase 7** | P7 | 可视化面板 | GradCAM/特征图接口 + 前端展示 | 2–3 天 |
| **Phase 8** | P8 | 集成测试 | 端到端测试，Bug 修复 | 2 天 |

**总估算：约 19–26 个工作日**

---

## 三、目录结构规划

```
Enneuro-Project/
├── code/
│   ├── eneuro/                    # 框架核心（尽量不改动）
│   ├── serving/                   # 现有推理服务（保留，扩展）
│   └── web_server/                # 新增 Web API 服务
│       ├── __init__.py
│       ├── main.py                # FastAPI app 入口
│       ├── routers/
│       │   ├── models.py          # /api/models 路由
│       │   ├── datasets.py        # /api/datasets 路由
│       │   ├── training.py        # /api/training 路由（含 SSE）
│       │   └── explain.py         # /api/explain 路由
│       ├── services/
│       │   ├── model_registry.py  # 层注册表 + 模型实例化
│       │   ├── task_manager.py    # 异步训练任务管理
│       │   ├── dataset_manager.py # 数据集文件管理
│       │   └── explain_service.py # GradCAM / 特征图服务
│       ├── schemas.py             # Pydantic 请求/响应模型
│       └── config.py              # 配置
└── frontend/                      # 新增 Vue 3 前端
    ├── index.html
    ├── vite.config.ts
    ├── package.json
    └── src/
        ├── main.ts
        ├── App.vue
        ├── router/
        ├── stores/                # Pinia 状态
        ├── api/                   # HTTP 封装
        └── views/
            ├── ModelDesigner.vue  # 模型设计器
            ├── DatasetManager.vue # 数据集管理
            ├── TrainingConsole.vue# 训练控制台
            └── VisualPanel.vue    # 可视化面板
```

---

## 四、各阶段详细计划

### Phase 1：后端基础服务

**目标：** 建立可运行的 FastAPI 应用骨架，实现层注册表和模型 API。

**任务清单：**

| # | 任务 | 文件 | 说明 |
|---|---|---|---|
| 1.1 | 初始化 FastAPI app，配置 CORS | `web_server/main.py` | 允许前端跨域访问 |
| 1.2 | 建立层注册表 | `web_server/services/model_registry.py` | 白名单类名 → 层实例化 |
| 1.3 | 实现模型配置 JSON Schema | `web_server/schemas.py` | 前端提交的层配置格式 |
| 1.4 | 实现 `POST /api/models/build` | `web_server/routers/models.py` | 从 JSON 配置构建模型并保存 |
| 1.5 | 实现 `GET /api/models/presets` | `web_server/routers/models.py` | 返回预置模型列表 |
| 1.6 | 实现 `GET /api/models/{model_id}` | `web_server/routers/models.py` | 返回模型结构 JSON |

---

### Phase 2：训练异步化 + SSE 推送

**目标：** 将 Trainer 改造为可在子进程运行、并通过 Queue 推送指标的模式。

**任务清单：**

| # | 任务 | 文件 | 说明 |
|---|---|---|---|
| 2.1 | 实现 `TaskManager` | `web_server/services/task_manager.py` | dict 管理 task_id → {process, queue, status} |
| 2.2 | 编写训练进程入口函数 | `web_server/services/task_manager.py` | `run_training_process(config, queue)` |
| 2.3 | 改造 Trainer 支持回调 | `code/eneuro/train/trainer.py` | 添加 `on_epoch_end` 回调参数 |
| 2.4 | 实现 `POST /api/training/start` | `web_server/routers/training.py` | 启动训练进程，返回 task_id |
| 2.5 | 实现 `GET /api/training/{task_id}/stream` | `web_server/routers/training.py` | SSE 端点，持续推送指标 |
| 2.6 | 实现 `POST /api/training/{task_id}/stop` | `web_server/routers/training.py` | 向训练进程发送停止信号 |
| 2.7 | 实现 `GET /api/training/{task_id}/status` | `web_server/routers/training.py` | 返回当前训练状态 |

---

### Phase 3：前端基础框架

**目标：** 搭建 Vue 3 + Vite 项目，实现路由和基础布局。

**任务清单：**

| # | 任务 | 说明 |
|---|---|---|
| 3.1 | 初始化 Vite + Vue 3 + TypeScript 项目 | `npm create vite@latest frontend` |
| 3.2 | 安装依赖 | `vue-router`, `pinia`, `echarts`, `vue-flow`, `axios` |
| 3.3 | 配置 Vite 反向代理 | 将 `/api` 请求代理到 FastAPI（dev 模式） |
| 3.4 | 实现全局布局（侧边导航 + 主内容区） | `App.vue` + `layouts/` |
| 3.5 | 配置路由 | 4 个视图页面的路由 |
| 3.6 | 封装 API 客户端 | `src/api/index.ts`（fetch + EventSource） |

---

### Phase 4：模型设计器

**目标：** 实现前端拖拽式模型架构设计，生成 JSON 并提交后端构建模型。

**任务清单：**

| # | 任务 | 说明 |
|---|---|---|
| 4.1 | 实现层节点组件 | 每种层类型一个节点，显示超参数 |
| 4.2 | 实现层配置面板 | 点击节点弹出参数配置表单 |
| 4.3 | 实现连接线（DAG 编辑） | Vue Flow 默认支持，需限制为链式连接 |
| 4.4 | 实现"预置模型"选择器 | 调用 `GET /api/models/presets` 填充下拉 |
| 4.5 | 实现提交按钮，调用 `POST /api/models/build` | 将节点图转为层配置 JSON |
| 4.6 | 展示当前已构建模型列表 | 调用 `GET /api/models` |

---

### Phase 5：数据集管理

**目标：** 支持本地数据集目录注册和简单文件上传，前端展示数据集信息。

**任务清单：**

| # | 任务 | 文件 | 说明 |
|---|---|---|---|
| 5.1 | 实现 `DatasetManager` | `web_server/services/dataset_manager.py` | 管理上传目录，记录数据集元数据 |
| 5.2 | 实现 `POST /api/datasets/upload` | `web_server/routers/datasets.py` | 接收 multipart 上传 |
| 5.3 | 实现 `POST /api/datasets/register` | `web_server/routers/datasets.py` | 注册服务器本地路径 |
| 5.4 | 实现 `GET /api/datasets` | `web_server/routers/datasets.py` | 返回数据集列表 |
| 5.5 | 实现 `GET /api/datasets/{id}/preview` | `web_server/routers/datasets.py` | 返回前 N 个样本的 base64 图像 |
| 5.6 | 前端数据集管理页面 | `DatasetManager.vue` | 上传、列表、预览 |

---

### Phase 6：训练控制台

**目标：** 前端实现训练配置、启动、停止，并实时绘制训练曲线。

**任务清单：**

| # | 任务 | 说明 |
|---|---|---|
| 6.1 | 训练配置表单 | 选择模型、数据集、epoch、lr、优化器等 |
| 6.2 | 调用 `POST /api/training/start` | 得到 task_id |
| 6.3 | 建立 SSE 连接 `EventSource` | 监听训练指标事件 |
| 6.4 | ECharts 实时折线图 | 动态追加 train_loss、val_loss、train_acc、val_acc |
| 6.5 | 停止按钮 | 调用 `POST /api/training/{task_id}/stop` |
| 6.6 | 状态标签 | running / stopped / finished / error |

---

### Phase 7：可视化面板

**目标：** 支持对已训练模型执行 GradCAM 和全通道特征图推理，前端展示结果。

**任务清单：**

| # | 任务 | 文件 | 说明 |
|---|---|---|---|
| 7.1 | 实现 `ExplainService` | `web_server/services/explain_service.py` | 封装 GradCAM 和特征图抓取 |
| 7.2 | 实现 `POST /api/explain/gradcam` | `web_server/routers/explain.py` | 返回 base64 热力图 |
| 7.3 | 实现 `POST /api/explain/feature_maps` | `web_server/routers/explain.py` | 返回指定层所有通道 base64 图 |
| 7.4 | 前端可视化面板 | `VisualPanel.vue` | 图像上传、层选择、结果展示网格 |

---

### Phase 8：集成测试

**目标：** 端到端验证所有功能，修复发现的 Bug。

**测试场景：**

1. 在模型设计器中构建一个 3 层 CNN，提交并获得 model_id
2. 上传 Donkey vs Horse 数据集（现有测试数据）
3. 在训练控制台启动训练，观察 SSE 曲线实时更新
4. 训练结束后，上传测试图片，生成 GradCAM 热力图
5. 生成指定 Conv 层的全通道特征图

---

## 五、接口规范摘要

### 模型相关

```
GET  /api/models/presets          → PresetModel[]
POST /api/models/build            → {model_id: str}
GET  /api/models/{model_id}       → ModelConfig
DELETE /api/models/{model_id}     → {ok: bool}
```

### 数据集相关

```
POST /api/datasets/upload         → {dataset_id: str}
POST /api/datasets/register       → {dataset_id: str}
GET  /api/datasets                → DatasetInfo[]
GET  /api/datasets/{id}/preview   → {samples: base64[]}
```

### 训练相关

```
POST /api/training/start          → {task_id: str}
GET  /api/training/{id}/stream    → SSE stream (text/event-stream)
POST /api/training/{id}/stop      → {ok: bool}
GET  /api/training/{id}/status    → TrainingStatus
```

SSE 事件格式：
```
event: epoch
data: {"epoch": 5, "train_loss": 0.32, "val_loss": 0.41, "train_acc": 0.88, "val_acc": 0.85}

event: done
data: {"status": "finished", "best_val_acc": 0.91}

event: error
data: {"message": "CUDA out of memory"}
```

### 可视化相关

```
POST /api/explain/gradcam         → {heatmap: base64_png, overlay: base64_png}
POST /api/explain/feature_maps    → {maps: [{channel: int, image: base64_png}]}
```

---

## 六、依赖清单

### 新增后端依赖

```
fastapi>=0.110.0    # 已有
uvicorn[standard]   # 已有
python-multipart    # 文件上传
aiofiles            # 异步文件 IO
pillow              # 图像编码为 PNG/base64
```

### 前端依赖

```json
{
  "dependencies": {
    "vue": "^3.4",
    "vue-router": "^4.3",
    "pinia": "^2.1",
    "echarts": "^5.5",
    "@vue-flow/core": "^1.38",
    "axios": "^1.7"
  },
  "devDependencies": {
    "vite": "^5.0",
    "@vitejs/plugin-vue": "^5.0",
    "typescript": "^5.0"
  }
}
```

---

*文档结束*
