from pathlib import Path
from typing import Any
from ..utils import StateDict
from ..nn.module import Module
from ..nn.optim import Optimizer

class Serializer:
    @staticmethod
    def save(state: StateDict, path: str | Path) -> None:
        pass
    @staticmethod
    def load(state: StateDict, path: str | Path) -> None:
        pass

    @staticmethod
    def save_checkpoint(model: Module,
                    optimizer: Optimizer,
                    epoch: int,
                    path: str):
        pass
        """
        把 model/optimizer/scheduler/epoch 打包成纯 dict，
        调用 Serializer.save 写入磁盘。
        """

    @staticmethod
    def load_checkpoint(path: str,
                    model: Module,
                    optimizer: Optimizer):
        pass
        """
        反向恢复，返回 epoch；保证 model/optimizer/scheduler 状态与保存时一致。
        """