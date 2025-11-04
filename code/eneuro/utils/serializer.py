from pathlib import Path
from typing import Any
from ..utils import StateDict
from ..nn.module import Module
from ..nn.optim import Optimizer
import json

class Serializer:
    @staticmethod
    def save(state: StateDict, path: str | Path) -> None:
        data = {}
        d = state.to_dict()

        if state is Module:
            data.update(model_state=d)
        elif state is Optimizer:
            data.update(optim_state=d)
        else:
            data = d
            print("Warning: Unstandard StateDict Saved!")

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def load(state: StateDict, path: str | Path) -> None:
        with open(path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        data = dict(loaded)
        d = {}

        if state is Module:
            d = data.get('model_state',{})
        elif state is Optimizer:
            d = data.get('optim_state',{})
        else:
            d = data
            print("Warning: Unstandard StateDict Loaded!")

        state.from_dict(d)

    @staticmethod
    def save_checkpoint(model: Module,
                    optimizer: Optimizer,
                    epoch: int,
                    path: str):
        """
        把 model/optimizer/scheduler/epoch 打包成纯 dict，
        写入磁盘。
        """
        model_state = model.to_dict()
        optim_state = optimizer.to_dict()
        
        data: dict = {}
        data.update(model_state=model_state, optim_state=optim_state, epoch=epoch, path=path)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        

    @staticmethod
    def load_checkpoint(path: str,
                    model: Module,
                    optimizer: Optimizer) -> int:
        """
        反向恢复，返回 epoch；保证 model/optimizer/scheduler 状态与保存时一致。
        """
        with open(path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        data = dict(loaded)

        model_state = data.get('model_state',{})
        optim_state = data.get('optim_state',{})
        epoch = data.get('epoch', 0)

        model.from_dict(model_state)
        optimizer.from_dict(optim_state)
        return epoch
