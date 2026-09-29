from pathlib import Path
from ..utils import StateDict
from ..nn.module import Module
from ..nn.optim import Optimizer
from ..base import Tensor
from ..base.functions import get_array_module
import json
import inspect
import os
import pickle

import numpy as np

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None


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
        if path.suffix.lower() in {'.pkl', '.pickle', '.bin'}:
            return Serializer.save_checkpoint_binary(model, optimizer, epoch, path,
                                                     model_config=model_config,
                                                     model_id=model_id)
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
    def save_checkpoint_binary(model: Module,
                               optimizer: Optimizer,
                               epoch: int,
                               path: str | Path,
                               model_config: dict = None,
                               model_id: str = None) -> None:
        """Save a compact binary checkpoint without converting arrays to JSON lists.

        The legacy JSON format is retained for interoperability, while the
        binary format is intended for GPU training checkpoints.  Arrays are
        copied to CPU NumPy storage before pickling, so the file is portable
        across CUDA devices and does not retain a CuPy memory pool reference.
        A temporary file plus ``os.replace`` prevents an interrupted save from
        corrupting the previous completed checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def cpu_array(value):
            if cp is not None and isinstance(value, cp.ndarray):
                return cp.asnumpy(value)
            if isinstance(value, np.ndarray):
                return np.asarray(value)
            return value

        def copy_tree(value):
            if isinstance(value, dict):
                return {key: copy_tree(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return type(value)(copy_tree(item) for item in value)
            return cpu_array(value)

        params = {}
        flat = {}
        model._flatten_params(flat)
        for key, param in flat.items():
            params[key] = {
                'data': cpu_array(param.data),
                # Epoch checkpoints are written after optimizer.step(), so
                # gradients are transient and are recomputed before the next
                # batch.  Omitting them keeps GPU checkpoints compact and
                # avoids serializing stale activation-sized float64 buffers.
                'grad': None,
                'requires_grad': param.requires_grad,
                'name': param.name,
            }

        payload = {
            'format': 'eneuro.binary_checkpoint.v1',
            'model_type': type(model).__name__,
            'model_state': {
                'metadata': getattr(model, 'metadata', {}),
                'params': params,
                'training': getattr(model, 'training', True),
                'model_class': type(model).__name__,
            },
            'optim_state': copy_tree(getattr(optimizer, '_state', {})),
            'epoch': int(epoch),
        }
        if model_id is not None:
            payload['model_id'] = model_id
        if model_config is not None:
            payload['model_config'] = model_config

        tmp_path = path.with_name(path.name + '.tmp')
        with open(tmp_path, 'wb') as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)

        if model_config is None:
            model_config = _auto_model_config(model)
        if model_config is not None:
            cfg_data = {
                'model_type': type(model).__name__,
                'model_config': model_config,
            }
            if model_id is not None:
                cfg_data['model_id'] = model_id
            with open(_config_path(path), 'w', encoding='utf-8') as handle:
                json.dump(cfg_data, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def load_checkpoint(path: str,
                        model: Module,
                        optimizer: Optimizer) -> int:
        """加载断点，返回已训练 epoch 数。"""
        path_obj = Path(path)
        if path_obj.suffix.lower() in {'.pkl', '.pickle', '.bin'}:
            return Serializer.load_checkpoint_binary(path_obj, model, optimizer)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model.from_dict(data.get('model_state', {}))
        optimizer.from_dict(data.get('optim_state', {}))
        return data.get('epoch', 0)

    @staticmethod
    def load_checkpoint_binary(path: str | Path,
                               model: Module,
                               optimizer: Optimizer) -> int:
        """Load a checkpoint produced by :meth:`save_checkpoint_binary`."""
        with open(path, 'rb') as handle:
            payload = pickle.load(handle)

        flat = {}
        model._flatten_params(flat)
        for key, saved in payload.get('model_state', {}).get('params', {}).items():
            if key not in flat:
                continue
            param = flat[key]
            if saved.get('data') is None:
                param.data = None
                param.grad = None
                param.requires_grad = saved.get('requires_grad', False)
                param.name = saved.get('name')
                continue
            if param.data is None:
                existing = next(
                    (item.data for item in flat.values() if item.data is not None),
                    None,
                )
                xp = get_array_module(existing) if existing is not None else np
                target_dtype = np.asarray(saved['data']).dtype
            else:
                xp = get_array_module(param.data)
                target_dtype = param.data.dtype
            param.data = xp.asarray(saved['data'], dtype=target_dtype)
            if saved.get('grad') is None:
                param.grad = None
            else:
                param.grad = Tensor(
                    xp.asarray(saved['grad'], dtype=param.data.dtype),
                    requires_grad=False,
                )
            param.requires_grad = saved.get('requires_grad', False)
            param.name = saved.get('name')

        state = payload.get('optim_state', {})
        # Restore optimizer arrays on the same backend as the model.  Keeping
        # them as CPU NumPy arrays would force every Adam step to copy each
        # momentum buffer back to the GPU after a resume.
        backend_sample = next(
            (item.data for item in flat.values() if item.data is not None),
            None,
        )
        xp = get_array_module(backend_sample) if backend_sample is not None else np

        def restore_tree(value):
            if isinstance(value, dict):
                return {key: restore_tree(item) for key, item in value.items()}
            if isinstance(value, np.ndarray):
                return xp.asarray(value)
            return value

        optimizer._state = restore_tree(state)
        if hasattr(model, 'training'):
            model.training = payload.get('model_state', {}).get(
                'training', getattr(model, 'training', True)
            )
        return int(payload.get('epoch', 0))
