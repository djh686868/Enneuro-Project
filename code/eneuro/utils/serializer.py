from pathlib import Path
from ..utils import StateDict
from ..nn.module import Module
from ..nn.optim import Optimizer
import json
import inspect


def _config_path(weights_path: Path) -> Path:
    """foo.json → foo.config.json"""
    return weights_path.with_name(weights_path.stem + '.config.json')


def _auto_model_config(model: Module) -> dict | None:
    """
    尝试从模型构造函数签名和实例属性中自动推导 model_config。
    规则：遍历 __init__ 的非 self 参数，若模型实例有同名属性则取其值。
    推导失败时返回 None（不报错）。
    """
    try:
        sig = inspect.signature(type(model).__init__)
        params = {}
        for name, param in sig.parameters.items():
            if name == 'self':
                continue
            if hasattr(model, name):
                params[name] = getattr(model, name)
            elif param.default is not inspect.Parameter.empty:
                params[name] = param.default
            else:
                return None   # 有必填参数但找不到对应属性，放弃推导
        return {"type": type(model).__name__, "params": params}
    except Exception:
        return None


class Serializer:
    """
    统一 JSON 格式（与 Web 端保存/加载完全一致）。

    权重文件  foo.json        —— model_type + model_state（+ 可选 model_config/model_id）
    配置文件  foo.config.json —— model_type + model_config  （单独保存，供 Web 端导入架构）

    Web 端格式
    ──────────
    {
        "model_id":     "d06cc8b9",
        "model_type":   "ResNet18",
        "model_config": {"type": "ResNet18", "params": {"in_channels": 3, "num_classes": 10}},
        "model_state":  { ... }
    }
    """

    @staticmethod
    def save(model: Module,
             path: str | Path,
             model_config: dict = None,
             model_id: str = None) -> None:
        """
        保存模型权重，并将 model_config 写入同名 .config.json 文件。

        Parameters
        ----------
        model_config : {"type": "ResNet18", "params": {"in_channels": 3, "num_classes": 10}}
                       传入后会额外生成 <stem>.config.json，Web 端可直接导入架构或加载权重。
        """
        path = Path(path)
        data = {
            "model_type":  type(model).__name__,
            "model_state": model.to_dict(),
        }
        if model_id is not None:
            data["model_id"] = model_id
        if model_config is not None:
            data["model_config"] = model_config

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 未传入 model_config 时尝试自动推导
        if model_config is None:
            model_config = _auto_model_config(model)

        # 同时写出独立 config 文件
        if model_config is not None:
            cfg_data = {
                "model_type":   type(model).__name__,
                "model_config": model_config,
            }
            if model_id is not None:
                cfg_data["model_id"] = model_id
            cfg_path = _config_path(path)
            with open(cfg_path, 'w', encoding='utf-8') as f:
                json.dump(cfg_data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load(model: Module, path: str | Path) -> None:
        """从 JSON 文件加载权重（兼容 Web 端格式及旧格式）。"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model.from_dict(data.get('model_state', data))

    @staticmethod
    def save_checkpoint(model: Module,
                        optimizer: Optimizer,
                        epoch: int,
                        path: str,
                        model_config: dict = None,
                        model_id: str = None) -> None:
        """保存训练断点（模型 + 优化器 + epoch），同时写出独立 config 文件。"""
        path = Path(path)
        data = {
            "model_type":  type(model).__name__,
            "model_state": model.to_dict(),
            "optim_state": optimizer.to_dict(),
            "epoch":       epoch,
        }
        if model_id is not None:
            data["model_id"] = model_id
        if model_config is not None:
            data["model_config"] = model_config

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if model_config is None:
            model_config = _auto_model_config(model)

        if model_config is not None:
            cfg_data = {
                "model_type":   type(model).__name__,
                "model_config": model_config,
            }
            if model_id is not None:
                cfg_data["model_id"] = model_id
            with open(_config_path(path), 'w', encoding='utf-8') as f:
                json.dump(cfg_data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_checkpoint(path: str,
                        model: Module,
                        optimizer: Optimizer) -> int:
        """加载断点，返回已训练 epoch 数。"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model.from_dict(data.get('model_state', {}))
        optimizer.from_dict(data.get('optim_state', {}))
        return data.get('epoch', 0)
