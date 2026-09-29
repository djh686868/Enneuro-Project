"""层注册表：白名单类名 -> 层类，以及从 JSON 配置实例化模型的工厂。"""
import uuid
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from eneuro.nn.module import (
    Linear, Conv2d, BatchNorm2d, Sequential, MLP, CNNWithPooling,
    ResidualBlock, LeNet, AlexNet, VGG, ResNet18,
)
from eneuro.nn.module import Layer

# 尝试导入自定义模型（路径不在包内时跳过）
try:
    import importlib.util, os as _os
    _donkey_path = _os.path.join(_os.path.dirname(__file__),
                                 "../../../tests/test_donkey/model.py")
    _spec = importlib.util.spec_from_file_location("donkey_model", _donkey_path)
    _mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    ResNet18AutoDrive = _mod.ResNet18AutoDrive
except Exception:
    ResNet18AutoDrive = None

LAYER_REGISTRY: dict[str, type] = {
    "Linear":         Linear,
    "Conv2d":         Conv2d,
    "BatchNorm2d":    BatchNorm2d,
    "Sequential":     Sequential,
    "MLP":            MLP,
    "CNNWithPooling": CNNWithPooling,
    "ResidualBlock":  ResidualBlock,
    "LeNet":          LeNet,
    "AlexNet":        AlexNet,
    "VGG":            VGG,
    "ResNet18":       ResNet18,
    **({"ResNet18AutoDrive": ResNet18AutoDrive} if ResNet18AutoDrive else {}),
}

PRESET_MODELS = {
    # ── 基础模型 ──
    "MLP_3Layer": {
        "type": "MLP",
        "params": {"fc_output_sizes": [256, 128, 10]},
        "description": "3 层 MLP（256→128→10），适合展平后的向量输入",
        "recommended_input": "任意向量输入",
    },
    "SimpleCNN": {
        "type": "CNNWithPooling",
        "params": {"in_channels": 1, "num_classes": 10},
        "description": "2 层 Conv+Pool 简单 CNN，适合灰度图像分类",
        "recommended_input": "1 通道图像，如 32×32",
    },
    # ── 经典架构 ──
    "LeNet": {
        "type": "LeNet",
        "params": {"in_channels": 1, "num_classes": 10},
        "description": "LeNet-5 变体，经典手写字符识别网络（LeCun 1998）",
        "recommended_input": "1 通道图像，建议 ≥ 16×16",
    },
    "LeNet_RGB": {
        "type": "LeNet",
        "params": {"in_channels": 3, "num_classes": 10},
        "description": "LeNet-5 变体，RGB 三通道输入版本",
        "recommended_input": "3 通道图像，建议 ≥ 16×16",
    },
    "AlexNet": {
        "type": "AlexNet",
        "params": {"in_channels": 3, "num_classes": 10},
        "description": "AlexNet 轻量适配版（Krizhevsky 2012），3×3 卷积适配小输入",
        "recommended_input": "3 通道图像，建议 64×64 或以上",
    },
    "AlexNet_Gray": {
        "type": "AlexNet",
        "params": {"in_channels": 1, "num_classes": 10},
        "description": "AlexNet 灰度输入版本",
        "recommended_input": "1 通道图像，建议 64×64 或以上",
    },
    "VGG11": {
        "type": "VGG",
        "params": {"cfg": "VGG11", "in_channels": 3, "num_classes": 10},
        "description": "VGG-11（Simonyan & Zisserman 2014），8 个卷积层 + BN",
        "recommended_input": "3 通道图像，建议 ≥ 32×32",
    },
    "VGG13": {
        "type": "VGG",
        "params": {"cfg": "VGG13", "in_channels": 3, "num_classes": 10},
        "description": "VGG-13，比 VGG-11 多两个卷积层",
        "recommended_input": "3 通道图像，建议 ≥ 32×32",
    },
    "VGG16": {
        "type": "VGG",
        "params": {"cfg": "VGG16", "in_channels": 3, "num_classes": 10},
        "description": "VGG-16，13 个卷积层 + 3 个全连接层，深度特征提取",
        "recommended_input": "3 通道图像，建议 ≥ 32×32",
    },
    "ResNet18": {
        "type": "ResNet18",
        "params": {"in_channels": 3, "num_classes": 10},
        "description": "ResNet-18（He et al. 2016），8 个残差块 + 全局平均池化，3 通道",
        "recommended_input": "3 通道图像，建议 ≥ 32×32",
    },
    "ResNet18_Gray": {
        "type": "ResNet18",
        "params": {"in_channels": 1, "num_classes": 10},
        "description": "ResNet-18 灰度输入版本，1 通道",
        "recommended_input": "1 通道图像，建议 ≥ 32×32",
    },
    **( {"ResNet18AutoDrive": {
        "type": "ResNet18AutoDrive",
        "params": {"in_channels": 3, "num_classes": 1},
        "description": "ResNet-18 端到端自动驾驶（回归），7×7 标准 stem，输出方向角（num_classes=1）",
        "recommended_input": "3 通道图像，160×120",
    }} if ResNet18AutoDrive else {} ),
}

_model_store: dict[str, Layer] = {}
_model_configs: dict[str, dict] = {}


def build_model_from_config(config: dict) -> tuple[str, Layer]:
    if "preset" in config and config["preset"]:
        preset_name = config["preset"]
        if preset_name not in PRESET_MODELS:
            raise ValueError(f"Unknown preset: {preset_name}")
        preset = PRESET_MODELS[preset_name]
        cls = LAYER_REGISTRY[preset["type"]]
        model = cls(**preset["params"])
    elif "type" in config and config["type"]:
        # 直接类型格式：{"type": "ResNet18AutoDrive", "params": {...}}
        type_name = config["type"]
        if type_name not in LAYER_REGISTRY:
            raise ValueError(f"Unknown model type: {type_name}")
        cls = LAYER_REGISTRY[type_name]
        model = cls(**config.get("params", {}))
    elif "layers" in config and config["layers"]:
        layers = []
        for layer_cfg in config["layers"]:
            layer_type = layer_cfg["type"]
            if layer_type not in LAYER_REGISTRY:
                raise ValueError(f"Unknown layer type: {layer_type}")
            cls = LAYER_REGISTRY[layer_type]
            layers.append(cls(**layer_cfg.get("params", {})))
        model = Sequential(*layers)
    else:
        raise ValueError("Config must have 'preset', 'type', or 'layers'")

    model_id = str(uuid.uuid4())[:8]
    _model_store[model_id] = model
    _model_configs[model_id] = config
    return model_id, model


def get_model(model_id: str) -> Layer:
    if model_id not in _model_store:
        raise KeyError(f"Model {model_id} not found")
    return _model_store[model_id]


def get_model_config(model_id: str) -> dict:
    return _model_configs.get(model_id, {})


def list_models() -> list[dict]:
    return [
        {"model_id": mid, "type": type(m).__name__}
        for mid, m in _model_store.items()
    ]


def list_presets() -> list[dict]:
    return [{"name": k, **v} for k, v in PRESET_MODELS.items()]
