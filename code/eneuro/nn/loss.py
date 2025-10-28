from ..core import Tensor

class Loss:
    def __init__(self) -> None:
        pass

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        raise

class MSELoss(Loss):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return super().__call__(pred, target)