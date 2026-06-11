"""层注册表：白名单类名 -> 层类，以及从 JSON 配置实例化模型的工厂。"""
import uuid
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from eneuro.nn.module import (
    Linear, Conv2d, BatchNorm2d, Sequential, MLP, CNNWithPooling,
)
from eneuro.nn.module import Layer

LAYER_REGISTRY: dict[str, type] = {
    "Linear":         Linear,
    "Conv2d":         Conv2d,
    "BatchNorm2d":    BatchNorm2d,
    "Sequential":     Sequential,
    "MLP":            MLP,
    "CNNWithPooling": CNNWithPooling,
}

PRESET_MODELS = {
    "MLP_3Layer": {
        "type": "MLP",
        "params": {"fc_output_sizes": [256, 128, 10]},
    },
    "SimpleCNN": {
        "type": "CNNWithPooling",
        "params": {"in_channels": 1, "num_classes": 10},
    },
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
        raise ValueError("Config must have 'preset' or 'layers'")

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
