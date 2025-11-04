from ..nn.module import Module
from ..nn.loss import Loss
from ..nn.optim import Optimizer
from ..core import Tensor

class Trainer:
    def __init__(self) -> None:
        self.model: Module
        self.loss_fn: Loss
        self.optimizer: Optimizer

    def train_epoch(self, data: list[tuple[Tensor,Tensor]]):
        pass