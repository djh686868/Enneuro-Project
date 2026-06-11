# EnNeuro Web 端可视化 — 计划执行指导

> 文档版本：v1.0  
> 日期：2026-06-11  
> 本文档提供详细步骤与代码示例，可直接按顺序执行

---

## Phase 1：后端基础服务

### Step 1.1 — 安装额外后端依赖

```powershell
pip install fastapi uvicorn[standard] python-multipart aiofiles pillow
```

---

### Step 1.2 — 层注册表（`model_registry.py`）

新建文件 `code/web_server/services/model_registry.py`：

```python
"""层注册表：白名单类名 -> 层类，以及从 JSON 配置实例化模型的工厂。"""
import uuid
from typing import Any

# ---- 白名单注册 ----
from eneuro.nn.module import (
    Linear, Conv2d, Deconv2d, BatchNorm, BatchNorm2d,
    Sequential, MLP, CNNWithPooling, ResidualBlock,
)
from eneuro.nn.module import Layer

LAYER_REGISTRY: dict[str, type] = {
    "Linear":        Linear,
    "Conv2d":        Conv2d,
    "Deconv2d":      Deconv2d,
    "BatchNorm":     BatchNorm,
    "BatchNorm2d":   BatchNorm2d,
    "Sequential":    Sequential,
    "MLP":           MLP,
    "CNNWithPooling": CNNWithPooling,
    "ResidualBlock": ResidualBlock,
}

PRESET_MODELS = {
    "MLP_3Layer": {
        "type": "MLP",
        "params": {"sizes": [784, 256, 128, 10], "activation": "relu"}
    },
    "SimpleCNN": {
        "type": "CNNWithPooling",
        "params": {"in_channels": 1, "num_classes": 10}
    },
}

# ---- 内存中的模型存储 ----
_model_store: dict[str, Any] = {}


def build_model_from_config(config: dict) -> tuple[str, Layer]:
    """
    config 格式:
    {
      "layers": [
        {"type": "Conv2d", "params": {"out_channels": 32, "kernel_size": 3}},
        {"type": "BatchNorm2d", "params": {}},
        ...
      ]
    }
    或者预置:
    {"preset": "MLP_3Layer"}
    """
    if "preset" in config:
        preset_name = config["preset"]
        if preset_name not in PRESET_MODELS:
            raise ValueError(f"Unknown preset: {preset_name}")
        preset = PRESET_MODELS[preset_name]
        cls = LAYER_REGISTRY[preset["type"]]
        model = cls(**preset["params"])
    else:
        layers = []
        for layer_cfg in config["layers"]:
            layer_type = layer_cfg["type"]
            if layer_type not in LAYER_REGISTRY:
                raise ValueError(f"Unknown layer type: {layer_type}")
            cls = LAYER_REGISTRY[layer_type]
            layers.append(cls(**layer_cfg.get("params", {})))
        model = Sequential(*layers)

    model_id = str(uuid.uuid4())[:8]
    _model_store[model_id] = model
    return model_id, model


def get_model(model_id: str) -> Layer:
    if model_id not in _model_store:
        raise KeyError(f"Model {model_id} not found")
    return _model_store[model_id]


def list_models() -> list[dict]:
    result = []
    for mid, m in _model_store.items():
        result.append({"model_id": mid, "type": type(m).__name__})
    return result


def list_presets() -> list[dict]:
    return [{"name": k, **v} for k, v in PRESET_MODELS.items()]
```

---

### Step 1.3 — Pydantic Schema（`schemas.py`）

新建文件 `code/web_server/schemas.py`：

```python
from pydantic import BaseModel
from typing import Any, Optional


class LayerConfig(BaseModel):
    type: str
    params: dict[str, Any] = {}


class BuildModelRequest(BaseModel):
    """支持两种方式：layers 列表 或 preset 名称"""
    layers: Optional[list[LayerConfig]] = None
    preset: Optional[str] = None


class BuildModelResponse(BaseModel):
    model_id: str
    model_type: str


class DatasetRegisterRequest(BaseModel):
    path: str          # 服务器本地路径
    name: str
    description: str = ""


class TrainingStartRequest(BaseModel):
    model_id: str
    dataset_id: str
    epochs: int = 10
    batch_size: int = 32
    lr: float = 0.001
    optimizer: str = "adam"   # "sgd" | "adam" | "momentum_sgd"
    device: str = "cpu"       # "cpu" | "cuda"
    val_split: float = 0.2


class GradCAMRequest(BaseModel):
    model_id: str
    image_base64: str          # PNG base64 编码的输入图像
    target_layer_name: str     # 例如 "conv1" 或层索引 "layer_2"
    class_idx: Optional[int] = None


class FeatureMapsRequest(BaseModel):
    model_id: str
    image_base64: str
    target_layer_name: str
```

---

### Step 1.4 — 模型路由（`routers/models.py`）

新建文件 `code/web_server/routers/models.py`：

```python
from fastapi import APIRouter, HTTPException
from web_server.schemas import BuildModelRequest, BuildModelResponse
from web_server.services.model_registry import (
    build_model_from_config, list_models, list_presets, get_model
)

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/presets")
def get_presets():
    return list_presets()


@router.get("")
def get_models():
    return list_models()


@router.post("/build", response_model=BuildModelResponse)
def build_model(req: BuildModelRequest):
    config = req.model_dump(exclude_none=True)
    # 将 Pydantic 格式转为注册表期望格式
    if "layers" in config:
        config["layers"] = [
            {"type": l["type"], "params": l["params"]}
            for l in config["layers"]
        ]
    try:
        model_id, model = build_model_from_config(config)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return BuildModelResponse(model_id=model_id, model_type=type(model).__name__)


@router.get("/{model_id}")
def get_model_info(model_id: str):
    try:
        model = get_model(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"model_id": model_id, "type": type(model).__name__}
```

---

### Step 1.5 — FastAPI 主入口（`main.py`）

新建文件 `code/web_server/main.py`：

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os

from web_server.routers import models, datasets, training, explain

app = FastAPI(title="EnNeuro Web API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 开发时放开；生产环境限制为前端域名
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(models.router)
app.include_router(datasets.router)
app.include_router(training.router)
app.include_router(explain.router)

# 如果 frontend/dist 存在则挂载静态文件（生产部署）
_dist = os.path.join(os.path.dirname(__file__), "../../frontend/dist")
if os.path.exists(_dist):
    app.mount("/", StaticFiles(directory=_dist, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("web_server.main:app", host="0.0.0.0", port=8000, reload=True)
```

**启动命令：**
```powershell
# 在 code/ 目录下执行
cd code
python -m web_server.main
# 或
uvicorn web_server.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Phase 2：训练异步化 + SSE 推送

### Step 2.1 — 改造 Trainer 支持回调

编辑 `code/eneuro/train/trainer.py`，在 `Trainer.__init__` 中添加 `on_epoch_end` 参数，并在每个 epoch 结束时调用：

```python
class Trainer:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        visualizer=None,
        enable_early_stop=False,
        on_epoch_end=None,   # 新增：epoch 结束回调
    ):
        # ... 原有初始化 ...
        self._on_epoch_end = on_epoch_end
        self._stop_requested = False  # 新增：停止信号

    def request_stop(self):
        """从外部调用以请求停止训练。"""
        self._stop_requested = True

    def train(self, train_loader, val_loader, epochs, ...):
        for epoch in range(1, epochs + 1):
            # ... 原有训练逻辑 ...

            # epoch 结束后调用回调
            if self._on_epoch_end is not None:
                self._on_epoch_end({
                    "epoch": epoch,
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "train_acc": float(train_acc),
                    "val_acc": float(val_acc),
                })

            # 检查停止信号
            if self._stop_requested:
                break
```

---

### Step 2.2 — 任务管理器（`task_manager.py`）

新建文件 `code/web_server/services/task_manager.py`：

```python
"""
训练任务管理器。
每个训练任务在独立子进程中运行，通过 multiprocessing.Queue 将指标推送回主进程。
"""
import uuid
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Optional
import sys, os

# 确保 eneuro 包可导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))


@dataclass
class TrainingTask:
    task_id: str
    process: Optional[mp.Process] = None
    queue: Optional[mp.Queue] = None
    status: str = "pending"   # pending | running | finished | stopped | error
    metrics_history: list = field(default_factory=list)


_tasks: dict[str, TrainingTask] = {}


def _training_worker(config: dict, queue: mp.Queue):
    """子进程入口：实例化模型、加载数据、运行训练。"""
    import numpy as np
    from eneuro.nn.module import Sequential
    from eneuro.train.trainer import Trainer
    from eneuro.nn.optim import Adam, SGD, MomentumSGD
    from eneuro.nn.loss import SoftmaxWithLoss

    # 重新从注册表构建模型（跨进程不能共享对象）
    from web_server.services.model_registry import build_model_from_config
    from web_server.services.dataset_manager import load_dataset_for_training

    try:
        queue.put({"type": "status", "status": "running"})

        model_config = config["model_config"]
        _, model = build_model_from_config(model_config)

        train_loader, val_loader = load_dataset_for_training(
            dataset_id=config["dataset_id"],
            batch_size=config["batch_size"],
            val_split=config["val_split"],
        )

        optim_map = {"adam": Adam, "sgd": SGD, "momentum_sgd": MomentumSGD}
        optimizer = optim_map[config["optimizer"]](model.params(), lr=config["lr"])
        loss_fn = SoftmaxWithLoss()

        def on_epoch_end(metrics: dict):
            queue.put({"type": "epoch", **metrics})

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            on_epoch_end=on_epoch_end,
        )
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            device=config["device"],
        )
        queue.put({"type": "done", "status": "finished"})

    except Exception as e:
        queue.put({"type": "error", "message": str(e)})


def start_training(training_config: dict, model_config: dict) -> str:
    task_id = str(uuid.uuid4())[:8]
    queue = mp.Queue()
    full_config = {**training_config, "model_config": model_config}

    proc = mp.Process(
        target=_training_worker,
        args=(full_config, queue),
        daemon=True,
    )
    proc.start()

    task = TrainingTask(task_id=task_id, process=proc, queue=queue)
    _tasks[task_id] = task
    return task_id


def stop_training(task_id: str) -> bool:
    task = _tasks.get(task_id)
    if task is None or task.process is None:
        return False
    task.process.terminate()
    task.status = "stopped"
    return True


def get_task(task_id: str) -> Optional[TrainingTask]:
    return _tasks.get(task_id)


def drain_queue(task_id: str) -> list[dict]:
    """读取队列中所有当前可用消息（非阻塞）。"""
    task = _tasks.get(task_id)
    if task is None:
        return []
    messages = []
    while not task.queue.empty():
        msg = task.queue.get_nowait()
        messages.append(msg)
        if msg["type"] in ("done", "error"):
            task.status = msg.get("status", msg["type"])
    return messages
```

---

### Step 2.3 — 训练路由（SSE 端点）

新建文件 `code/web_server/routers/training.py`：

```python
import asyncio
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from web_server.schemas import TrainingStartRequest
from web_server.services.task_manager import (
    start_training, stop_training, get_task, drain_queue
)
from web_server.services.model_registry import get_model, _model_store

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start")
def start_training_task(req: TrainingStartRequest):
    # 验证 model_id
    if req.model_id not in _model_store:
        raise HTTPException(status_code=404, detail="Model not found")

    # 将模型重建配置传给子进程（子进程无法直接共享模型对象）
    model_config = {"preset": None, "model_id": req.model_id}

    training_config = {
        "dataset_id": req.dataset_id,
        "epochs": req.epochs,
        "batch_size": req.batch_size,
        "lr": req.lr,
        "optimizer": req.optimizer,
        "device": req.device,
        "val_split": req.val_split,
    }

    task_id = start_training(training_config, model_config)
    return {"task_id": task_id}


@router.get("/{task_id}/stream")
async def stream_training(task_id: str):
    """SSE 端点：持续推送训练指标，直到训练结束。"""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        while True:
            messages = drain_queue(task_id)
            for msg in messages:
                event_type = msg.pop("type")
                yield f"event: {event_type}\ndata: {json.dumps(msg)}\n\n"
                if event_type in ("done", "error"):
                    return
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{task_id}/stop")
def stop_task(task_id: str):
    ok = stop_training(task_id)
    return {"ok": ok}


@router.get("/{task_id}/status")
def task_status(task_id: str):
    task = get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"task_id": task_id, "status": task.status}
```

---

### Step 2.4 — 数据集管理器（`dataset_manager.py`）

新建文件 `code/web_server/services/dataset_manager.py`：

```python
"""数据集管理：注册、存储路径、构建 DataLoader。"""
import os
import uuid
import numpy as np
from eneuro.data.dataset import Dataset
from eneuro.data.dataloader import DataLoader

_datasets: dict[str, dict] = {}
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "../../../uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def register_dataset(name: str, path: str, description: str = "") -> str:
    dataset_id = str(uuid.uuid4())[:8]
    _datasets[dataset_id] = {
        "dataset_id": dataset_id,
        "name": name,
        "path": path,
        "description": description,
    }
    return dataset_id


def list_datasets() -> list[dict]:
    return list(_datasets.values())


class FolderImageDataset(Dataset):
    """
    从目录结构加载图像分类数据集。
    期望目录格式：
        root/
          class_a/  image1.png, image2.jpg, ...
          class_b/  ...
    """
    def __init__(self, root: str, img_size=(32, 32), transform=None):
        self.root = root
        self.img_size = img_size
        self._samples: list[tuple[str, int]] = []
        self._classes: list[str] = []
        super().__init__(transform=transform)

    def prepare(self):
        import cv2
        self._classes = sorted([
            d for d in os.listdir(self.root)
            if os.path.isdir(os.path.join(self.root, d))
        ])
        for label, cls in enumerate(self._classes):
            cls_dir = os.path.join(self.root, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    self._samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        import cv2
        path, label = self._samples[index]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.img_size)
        x = img.astype(np.float32) / 255.0
        x = x[np.newaxis, :, :]   # (1, H, W)
        return x, np.array(label, dtype=np.int32)


def load_dataset_for_training(dataset_id: str, batch_size: int, val_split: float):
    meta = _datasets.get(dataset_id)
    if meta is None:
        raise ValueError(f"Dataset {dataset_id} not found")

    dataset = FolderImageDataset(root=meta["path"])

    n = len(dataset)
    val_n = int(n * val_split)
    indices = np.random.permutation(n)
    val_idx = indices[:val_n].tolist()
    train_idx = indices[val_n:].tolist()

    class SubsetDataset(Dataset):
        def __init__(self, base, idx):
            self._base = base
            self._idx = idx
            super().__init__()
        def prepare(self): pass
        def __len__(self): return len(self._idx)
        def __getitem__(self, i): return self._base[self._idx[i]]

    train_ds = SubsetDataset(dataset, train_idx)
    val_ds = SubsetDataset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader
```

---

### Step 2.5 — 数据集路由（`routers/datasets.py`）

新建文件 `code/web_server/routers/datasets.py`：

```python
import os, base64, io
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File
import cv2

from web_server.schemas import DatasetRegisterRequest
from web_server.services.dataset_manager import (
    register_dataset, list_datasets, UPLOAD_DIR, _datasets
)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


@router.post("/register")
def register(req: DatasetRegisterRequest):
    if not os.path.exists(req.path):
        raise HTTPException(status_code=400, detail="Path does not exist")
    dataset_id = register_dataset(req.name, req.path, req.description)
    return {"dataset_id": dataset_id}


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """上传 zip 文件，解压到 uploads/ 目录，自动注册。"""
    import zipfile
    dest = os.path.join(UPLOAD_DIR, file.filename.replace(".zip", ""))
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(UPLOAD_DIR, file.filename)
    content = await file.read()
    with open(zip_path, "wb") as f:
        f.write(content)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    dataset_id = register_dataset(
        name=file.filename.replace(".zip", ""),
        path=dest,
    )
    return {"dataset_id": dataset_id}


@router.get("")
def get_datasets():
    return list_datasets()


@router.get("/{dataset_id}/preview")
def preview_dataset(dataset_id: str, n: int = 9):
    meta = _datasets.get(dataset_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    from web_server.services.dataset_manager import FolderImageDataset
    ds = FolderImageDataset(root=meta["path"])
    samples = []
    for i in range(min(n, len(ds))):
        img_arr, label = ds[i]   # (1, H, W)
        img_u8 = (img_arr[0] * 255).astype(np.uint8)
        _, buf = cv2.imencode(".png", img_u8)
        b64 = base64.b64encode(buf).decode()
        samples.append({"label": int(label), "image": b64})
    return {"samples": samples, "classes": ds._classes}
```

---

## Phase 3：前端项目初始化

### Step 3.1 — 创建 Vue 3 项目

```powershell
# 在项目根目录执行
npm create vite@latest frontend -- --template vue-ts
cd frontend
npm install
npm install vue-router@4 pinia echarts @vue-flow/core axios
npm install -D @types/node
```

### Step 3.2 — Vite 反向代理配置

编辑 `frontend/vite.config.ts`：

```typescript
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      }
    }
  }
})
```

### Step 3.3 — 路由配置

新建 `frontend/src/router/index.ts`：

```typescript
import { createRouter, createWebHistory } from 'vue-router'

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/designer' },
    { path: '/designer',  component: () => import('../views/ModelDesigner.vue') },
    { path: '/datasets',  component: () => import('../views/DatasetManager.vue') },
    { path: '/training',  component: () => import('../views/TrainingConsole.vue') },
    { path: '/visual',    component: () => import('../views/VisualPanel.vue') },
  ]
})
```

### Step 3.4 — API 客户端

新建 `frontend/src/api/index.ts`：

```typescript
const BASE = '/api'

export const api = {
  // 模型
  getPresets: () => fetch(`${BASE}/models/presets`).then(r => r.json()),
  getModels:  () => fetch(`${BASE}/models`).then(r => r.json()),
  buildModel: (body: object) =>
    fetch(`${BASE}/models/build`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => r.json()),

  // 数据集
  getDatasets:     () => fetch(`${BASE}/datasets`).then(r => r.json()),
  previewDataset:  (id: string, n = 9) =>
    fetch(`${BASE}/datasets/${id}/preview?n=${n}`).then(r => r.json()),
  registerDataset: (body: object) =>
    fetch(`${BASE}/datasets/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => r.json()),
  uploadDataset: (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return fetch(`${BASE}/datasets/upload`, { method: 'POST', body: fd }).then(r => r.json())
  },

  // 训练
  startTraining: (body: object) =>
    fetch(`${BASE}/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => r.json()),
  stopTraining: (taskId: string) =>
    fetch(`${BASE}/training/${taskId}/stop`, { method: 'POST' }).then(r => r.json()),
  getTrainingStatus: (taskId: string) =>
    fetch(`${BASE}/training/${taskId}/status`).then(r => r.json()),
  streamTraining: (taskId: string) => new EventSource(`${BASE}/training/${taskId}/stream`),

  // 可视化
  gradcam: (body: object) =>
    fetch(`${BASE}/explain/gradcam`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => r.json()),
  featureMaps: (body: object) =>
    fetch(`${BASE}/explain/feature_maps`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then(r => r.json()),
}
```

---

## Phase 4：模型设计器（核心前端视图）

新建 `frontend/src/views/ModelDesigner.vue`：

```vue
<template>
  <div class="designer">
    <aside class="layer-palette">
      <h3>层类型</h3>
      <div
        v-for="lt in layerTypes"
        :key="lt"
        class="layer-chip"
        @click="addLayer(lt)"
      >{{ lt }}</div>

      <h3 style="margin-top:16px">预置模型</h3>
      <div
        v-for="p in presets"
        :key="p.name"
        class="layer-chip preset"
        @click="loadPreset(p.name)"
      >{{ p.name }}</div>
    </aside>

    <main class="canvas">
      <VueFlow v-model="elements" @node-click="onNodeClick" fit-view-on-init>
        <Background />
        <Controls />
      </VueFlow>
    </main>

    <!-- 层参数配置面板 -->
    <aside v-if="selectedLayer" class="config-panel">
      <h3>配置：{{ selectedLayer.type }}</h3>
      <div v-for="(val, key) in selectedLayer.params" :key="key" class="param-row">
        <label>{{ key }}</label>
        <input v-model="selectedLayer.params[key]" />
      </div>
      <button @click="selectedLayer = null">关闭</button>
    </aside>

    <div class="toolbar">
      <button @click="submitModel" :disabled="!elements.length">构建模型</button>
      <span v-if="builtModelId">✓ model_id: {{ builtModelId }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { VueFlow, Background, Controls, useVueFlow } from '@vue-flow/core'
import '@vue-flow/core/dist/style.css'
import { api } from '../api'

const layerTypes = ['Conv2d', 'BatchNorm2d', 'Linear', 'ResidualBlock']
const presets = ref<any[]>([])
const elements = ref<any[]>([])
const selectedLayer = ref<any>(null)
const builtModelId = ref('')
let nodeCounter = 0

onMounted(async () => {
  presets.value = await api.getPresets()
})

function addLayer(type: string) {
  const id = `node-${nodeCounter++}`
  const defaultParams: Record<string, any> = {
    Conv2d:       { out_channels: 32, kernel_size: 3, padding: 1 },
    BatchNorm2d:  {},
    Linear:       { out_size: 128 },
    ResidualBlock: { in_channels: 32, out_channels: 32 },
  }
  elements.value.push({
    id,
    type: 'default',
    position: { x: 100, y: nodeCounter * 100 },
    data: { label: type, type, params: { ...(defaultParams[type] || {}) } },
  })
  // 自动连接到上一个节点
  if (elements.value.filter((e: any) => e.source === undefined).length > 1) {
    const nodes = elements.value.filter((e: any) => !e.source)
    const prev = nodes[nodes.length - 2]
    elements.value.push({
      id: `edge-${id}`,
      source: prev.id,
      target: id,
    })
  }
}

function loadPreset(name: string) {
  elements.value = []
  nodeCounter = 0
  // 清空画布，标记为 preset 提交
  elements.value.push({
    id: 'preset-node',
    type: 'default',
    position: { x: 200, y: 200 },
    data: { label: `[Preset] ${name}`, type: '__preset__', params: { preset: name } },
  })
}

function onNodeClick(_: any, node: any) {
  if (node.data.type !== '__preset__') {
    selectedLayer.value = node.data
  }
}

async function submitModel() {
  const nodes = elements.value.filter((e: any) => !e.source)
  let body: any
  if (nodes[0]?.data.type === '__preset__') {
    body = { preset: nodes[0].data.params.preset }
  } else {
    body = {
      layers: nodes.map((n: any) => ({ type: n.data.type, params: n.data.params }))
    }
  }
  const res = await api.buildModel(body)
  if (res.model_id) {
    builtModelId.value = res.model_id
  }
}
</script>

<style scoped>
.designer { display: flex; height: 100vh; gap: 0; }
.layer-palette { width: 160px; padding: 12px; background: #1a1a2e; overflow-y: auto; }
.layer-chip { padding: 6px 10px; margin: 4px 0; background: #16213e; border-radius: 6px;
              cursor: pointer; font-size: 13px; color: #e0e0e0; }
.layer-chip:hover { background: #0f3460; }
.preset { border-left: 3px solid #e94560; }
.canvas { flex: 1; height: 100%; }
.config-panel { width: 220px; padding: 16px; background: #1a1a2e; }
.param-row { display: flex; flex-direction: column; margin: 8px 0; }
.param-row label { font-size: 12px; color: #888; }
.param-row input { background: #0f3460; color: #fff; border: 1px solid #333;
                   border-radius: 4px; padding: 4px; }
.toolbar { position: fixed; bottom: 16px; right: 16px; display: flex;
           align-items: center; gap: 12px; }
button { padding: 8px 20px; background: #e94560; color: #fff; border: none;
         border-radius: 6px; cursor: pointer; font-size: 14px; }
button:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
```

---

## Phase 6：训练控制台（ECharts 实时曲线）

新建 `frontend/src/views/TrainingConsole.vue`：

```vue
<template>
  <div class="console">
    <section class="config-form">
      <h2>训练配置</h2>
      <label>Model ID</label>
      <input v-model="form.model_id" placeholder="从模型设计器复制" />
      <label>Dataset ID</label>
      <input v-model="form.dataset_id" placeholder="从数据集管理复制" />
      <label>Epochs</label>
      <input v-model.number="form.epochs" type="number" min="1" />
      <label>Batch Size</label>
      <input v-model.number="form.batch_size" type="number" />
      <label>Learning Rate</label>
      <input v-model.number="form.lr" type="number" step="0.0001" />
      <label>Optimizer</label>
      <select v-model="form.optimizer">
        <option value="adam">Adam</option>
        <option value="sgd">SGD</option>
        <option value="momentum_sgd">Momentum SGD</option>
      </select>
      <label>Device</label>
      <select v-model="form.device">
        <option value="cpu">CPU</option>
        <option value="cuda">CUDA</option>
      </select>
      <div class="btn-row">
        <button @click="startTraining" :disabled="status === 'running'">开始训练</button>
        <button @click="stopTraining" :disabled="status !== 'running'" class="stop-btn">停止</button>
      </div>
      <div class="status-badge" :class="status">{{ status }}</div>
    </section>

    <section class="charts">
      <div ref="lossChartEl" class="chart-box"></div>
      <div ref="accChartEl"  class="chart-box"></div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import { api } from '../api'

const form = ref({
  model_id: '', dataset_id: '',
  epochs: 10, batch_size: 32, lr: 0.001,
  optimizer: 'adam', device: 'cpu', val_split: 0.2,
})

const status = ref<string>('idle')
let taskId = ''
let es: EventSource | null = null

const lossChartEl = ref<HTMLElement>()
const accChartEl  = ref<HTMLElement>()
let lossChart: echarts.ECharts
let accChart:  echarts.ECharts

const lossData = { epochs: [] as number[], train: [] as number[], val: [] as number[] }
const accData  = { epochs: [] as number[], train: [] as number[], val: [] as number[] }

function makeOption(title: string, seriesNames: string[]) {
  return {
    title: { text: title, textStyle: { color: '#ccc' } },
    backgroundColor: '#1a1a2e',
    xAxis: { type: 'category', data: [], axisLine: { lineStyle: { color: '#555' } } },
    yAxis: { type: 'value', axisLine: { lineStyle: { color: '#555' } } },
    legend: { data: seriesNames, textStyle: { color: '#ccc' } },
    series: seriesNames.map(name => ({
      name, type: 'line', smooth: true, data: [],
      lineStyle: { width: 2 },
    })),
    grid: { left: '10%', right: '5%' },
  }
}

onMounted(() => {
  lossChart = echarts.init(lossChartEl.value!, 'dark')
  accChart  = echarts.init(accChartEl.value!, 'dark')
  lossChart.setOption(makeOption('Loss', ['Train Loss', 'Val Loss']))
  accChart.setOption(makeOption('Accuracy', ['Train Acc', 'Val Acc']))
})

onUnmounted(() => es?.close())

async function startTraining() {
  // 重置数据
  lossData.epochs = []; lossData.train = []; lossData.val = []
  accData.epochs  = []; accData.train  = []; accData.val  = []

  const res = await api.startTraining(form.value)
  taskId = res.task_id
  status.value = 'running'

  es = api.streamTraining(taskId)

  es.addEventListener('epoch', (e: MessageEvent) => {
    const d = JSON.parse(e.data)
    lossData.epochs.push(d.epoch)
    lossData.train.push(d.train_loss)
    lossData.val.push(d.val_loss)
    accData.epochs.push(d.epoch)
    accData.train.push(d.train_acc)
    accData.val.push(d.val_acc)

    lossChart.setOption({
      xAxis: { data: lossData.epochs },
      series: [{ data: lossData.train }, { data: lossData.val }],
    })
    accChart.setOption({
      xAxis: { data: accData.epochs },
      series: [{ data: accData.train }, { data: accData.val }],
    })
  })

  es.addEventListener('done',  (e: MessageEvent) => {
    status.value = 'finished'; es?.close()
  })
  es.addEventListener('error', (e: MessageEvent) => {
    status.value = 'error'; es?.close()
  })
}

async function stopTraining() {
  await api.stopTraining(taskId)
  es?.close()
  status.value = 'stopped'
}
</script>

<style scoped>
.console { display: flex; gap: 24px; padding: 20px; height: 100vh; }
.config-form { width: 260px; display: flex; flex-direction: column; gap: 6px;
               background: #1a1a2e; padding: 16px; border-radius: 10px; }
.config-form label { font-size: 12px; color: #aaa; }
.config-form input, .config-form select {
  background: #0f3460; color: #fff; border: 1px solid #333;
  border-radius: 4px; padding: 6px; font-size: 13px;
}
.btn-row { display: flex; gap: 8px; margin-top: 8px; }
button { flex: 1; padding: 8px; border: none; border-radius: 6px;
         cursor: pointer; font-size: 13px; background: #e94560; color: #fff; }
button:disabled { opacity: 0.4; cursor: not-allowed; }
.stop-btn { background: #555; }
.status-badge { text-align: center; padding: 4px 8px; border-radius: 4px; font-size: 12px;
                font-weight: bold; text-transform: uppercase; }
.running  { background: #1a6b40; color: #4ecca3; }
.finished { background: #1a3a6b; color: #6ab0f5; }
.stopped  { background: #555; color: #ccc; }
.error    { background: #6b1a1a; color: #f56a6a; }
.idle     { background: #333; color: #888; }
.charts { flex: 1; display: flex; flex-direction: column; gap: 16px; }
.chart-box { flex: 1; border-radius: 10px; overflow: hidden; }
</style>
```

---

## Phase 7：可视化服务与面板

### Step 7.1 — ExplainService（`services/explain_service.py`）

```python
"""GradCAM 和全通道特征图服务。"""
import base64
import io
import numpy as np
import cv2

from eneuro.explainability.gradcam import GradCAM
from eneuro.utils.hooks import capture_features
from web_server.services.model_registry import get_model


def _b64_png(arr_hw: np.ndarray) -> str:
    """将 (H, W) float[0,1] numpy 数组编码为 base64 PNG 字符串。"""
    img_u8 = (arr_hw * 255).clip(0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", img_u8)
    return base64.b64encode(buf).decode()


def _parse_image(b64_str: str) -> np.ndarray:
    """base64 PNG → (1, 1, H, W) float32 Tensor-ready numpy array。"""
    data = base64.b64decode(b64_str)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0
    return img[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)


def _find_layer(model, name: str):
    """按名称或索引查找层。"""
    # 先尝试属性名
    if hasattr(model, name):
        return getattr(model, name)
    # 再尝试 Sequential 索引（"layer_2" → layers[2]）
    if name.startswith("layer_") and hasattr(model, "layers"):
        idx = int(name.split("_")[1])
        return model.layers[idx]
    raise ValueError(f"Layer '{name}' not found in model")


def run_gradcam(model_id: str, image_b64: str, layer_name: str, class_idx=None):
    model = get_model(model_id)
    img_np = _parse_image(image_b64)

    from eneuro.base.core import Tensor
    x = Tensor(img_np, requires_grad=False)

    target_layer = _find_layer(model, layer_name)
    cam = GradCAM(model, target_layer)
    heatmap = cam.generate(x, class_idx=class_idx)   # (H, W) float[0,1]

    # 生成彩色叠加图
    h_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    orig_u8 = (img_np[0, 0] * 255).astype(np.uint8)
    orig_bgr = cv2.cvtColor(orig_u8, cv2.COLOR_GRAY2BGR)
    orig_bgr = cv2.resize(orig_bgr, (heatmap.shape[1], heatmap.shape[0]))
    overlay = cv2.addWeighted(orig_bgr, 0.5, h_color, 0.5, 0)

    _, buf_h = cv2.imencode(".png", h_color)
    _, buf_o = cv2.imencode(".png", overlay)
    return {
        "heatmap": base64.b64encode(buf_h).decode(),
        "overlay": base64.b64encode(buf_o).decode(),
    }


def run_feature_maps(model_id: str, image_b64: str, layer_name: str):
    model = get_model(model_id)
    img_np = _parse_image(image_b64)

    from eneuro.base.core import Tensor
    x = Tensor(img_np, requires_grad=False)

    target_layer = _find_layer(model, layer_name)
    features = {}
    hook = capture_features(target_layer)  # 注册前向钩子

    model(x)  # 前向推理以触发钩子

    acts = getattr(target_layer, "_captured_features", None)
    if acts is None:
        raise RuntimeError("No features captured — check layer name")

    if hasattr(acts, "data"):
        acts = acts.data
    if hasattr(acts, "get"):   # cupy
        acts = acts.get()

    # acts shape: (1, C, H, W) → 每通道独立图
    acts = acts[0]  # (C, H, W)
    maps = []
    for c in range(acts.shape[0]):
        ch = acts[c]
        ch_norm = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        maps.append({"channel": c, "image": _b64_png(ch_norm)})

    return {"maps": maps, "num_channels": len(maps)}
```

### Step 7.2 — 解释路由（`routers/explain.py`）

```python
from fastapi import APIRouter, HTTPException
from web_server.schemas import GradCAMRequest, FeatureMapsRequest
from web_server.services.explain_service import run_gradcam, run_feature_maps

router = APIRouter(prefix="/api/explain", tags=["explain"])


@router.post("/gradcam")
def gradcam(req: GradCAMRequest):
    try:
        return run_gradcam(
            model_id=req.model_id,
            image_b64=req.image_base64,
            layer_name=req.target_layer_name,
            class_idx=req.class_idx,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/feature_maps")
def feature_maps(req: FeatureMapsRequest):
    try:
        return run_feature_maps(
            model_id=req.model_id,
            image_b64=req.image_base64,
            layer_name=req.target_layer_name,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
```

### Step 7.3 — 可视化面板前端（`VisualPanel.vue`）

```vue
<template>
  <div class="visual-panel">
    <aside class="ctrl">
      <h2>可视化</h2>
      <label>Model ID</label>
      <input v-model="modelId" />
      <label>目标层名称</label>
      <input v-model="layerName" placeholder="如 conv1 或 layer_0" />
      <label>输入图像</label>
      <input type="file" accept="image/*" @change="onFile" />
      <img v-if="previewSrc" :src="previewSrc" class="preview-img" />
      <div class="btn-col">
        <button @click="doGradCAM" :disabled="!imgB64">GradCAM</button>
        <button @click="doFeatureMaps" :disabled="!imgB64">特征图</button>
      </div>
    </aside>

    <main class="results">
      <!-- GradCAM 结果 -->
      <section v-if="gradcamResult">
        <h3>GradCAM 热力图</h3>
        <div class="img-row">
          <figure>
            <img :src="`data:image/png;base64,${gradcamResult.heatmap}`" />
            <figcaption>热力图</figcaption>
          </figure>
          <figure>
            <img :src="`data:image/png;base64,${gradcamResult.overlay}`" />
            <figcaption>叠加图</figcaption>
          </figure>
        </div>
      </section>

      <!-- 特征图网格 -->
      <section v-if="featureMaps.length">
        <h3>全通道特征图（{{ featureMaps.length }} 通道）</h3>
        <div class="map-grid">
          <figure v-for="m in featureMaps" :key="m.channel">
            <img :src="`data:image/png;base64,${m.image}`" />
            <figcaption>Ch {{ m.channel }}</figcaption>
          </figure>
        </div>
      </section>
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { api } from '../api'

const modelId   = ref('')
const layerName = ref('')
const imgB64    = ref('')
const previewSrc = ref('')
const gradcamResult = ref<any>(null)
const featureMaps   = ref<any[]>([])

function onFile(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (!file) return
  const reader = new FileReader()
  reader.onload = ev => {
    const dataUrl = ev.target!.result as string
    previewSrc.value = dataUrl
    imgB64.value = dataUrl.split(',')[1]
  }
  reader.readAsDataURL(file)
}

async function doGradCAM() {
  featureMaps.value = []
  const res = await api.gradcam({
    model_id: modelId.value,
    image_base64: imgB64.value,
    target_layer_name: layerName.value,
  })
  gradcamResult.value = res
}

async function doFeatureMaps() {
  gradcamResult.value = null
  const res = await api.featureMaps({
    model_id: modelId.value,
    image_base64: imgB64.value,
    target_layer_name: layerName.value,
  })
  featureMaps.value = res.maps
}
</script>

<style scoped>
.visual-panel { display: flex; gap: 20px; padding: 20px; min-height: 100vh; }
.ctrl { width: 240px; display: flex; flex-direction: column; gap: 8px;
        background: #1a1a2e; padding: 16px; border-radius: 10px; }
.ctrl label { font-size: 12px; color: #aaa; }
.ctrl input[type=text], .ctrl input:not([type]) {
  background: #0f3460; color: #fff; border: 1px solid #333;
  border-radius: 4px; padding: 6px; font-size: 13px;
}
.preview-img { width: 100%; border-radius: 6px; margin-top: 4px; }
.btn-col { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
button { padding: 8px; background: #e94560; color: #fff; border: none;
         border-radius: 6px; cursor: pointer; }
button:disabled { opacity: 0.4; cursor: not-allowed; }
.results { flex: 1; overflow-y: auto; }
.img-row { display: flex; gap: 16px; }
.img-row figure, .map-grid figure { margin: 0; text-align: center; }
.img-row img { max-height: 300px; border-radius: 8px; }
figcaption { font-size: 12px; color: #888; margin-top: 4px; }
.map-grid { display: flex; flex-wrap: wrap; gap: 8px; }
.map-grid img { width: 80px; height: 80px; object-fit: cover; border-radius: 4px; }
</style>
```

---

## Phase 8：启动与验证

### 启动后端

```powershell
cd D:\Undergraduate\WZH\EnNeuro\tmp\Enneuro-Project\code
python -m web_server.main
# 输出：Uvicorn running on http://0.0.0.0:8000
```

### 启动前端开发服务器

```powershell
cd D:\Undergraduate\WZH\EnNeuro\tmp\Enneuro-Project\frontend
npm run dev
# 输出：VITE v5.x.x  ➜  Local: http://localhost:5173/
```

### 快速验证清单

```powershell
# 1. 健康检查
curl http://localhost:8000/api/models/presets

# 2. 构建预置模型
curl -X POST http://localhost:8000/api/models/build `
  -H "Content-Type: application/json" `
  -d '{"preset": "MLP_3Layer"}'
# 期望响应：{"model_id": "xxxx", "model_type": "MLP"}

# 3. 注册测试数据集（使用现有 donkey 数据）
curl -X POST http://localhost:8000/api/datasets/register `
  -H "Content-Type: application/json" `
  -d '{"path": "D:/Undergraduate/WZH/EnNeuro/tmp/Enneuro-Project/code/tests/test_donkey/data", "name": "donkey_horse"}'

# 4. 查看 API 文档
# 浏览器打开：http://localhost:8000/docs
```

### 端到端测试步骤（浏览器）

1. 打开 `http://localhost:5173/designer`
2. 点击 "SimpleCNN" 预置 → 点击"构建模型" → 记录 model_id
3. 切换到 `/datasets`，注册 donkey 数据集 → 记录 dataset_id
4. 切换到 `/training`，填入 model_id + dataset_id → 点击"开始训练"
5. 观察 ECharts 曲线实时更新（每 epoch 推送一次）
6. 训练完成后，切换到 `/visual`，上传一张 donkey 图片
7. 填入 model_id + 层名（如 `layer_0`） → 点击 GradCAM → 观察热力图

---

## 附录：常见问题

### Q: 训练子进程无法导入 eneuro 模块

在 `code/web_server/services/task_manager.py` 顶部确认已有：
```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
```

### Q: SSE 在浏览器中没有收到事件

检查 FastAPI response header：
```
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no   ← Nginx 需要此头
```

### Q: GradCAM 返回全黑热力图

参考 `doc/20260604_GradCAM热力图全零修复报告.md`。确认 `target_layer_name` 指向 Conv 层（不是 BN 层），框架内部会自动使用后继层梯度。

### Q: CUDA 版本错误

参考 `CLAUDE.md`：临时切换 CUDA 12.6：
```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;" + $env:PATH
```

---

*文档结束*
