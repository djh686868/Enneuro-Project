from ..core import Tensor

class Meter:
    def __init__(self) -> None:
        self.values: list[float]

    def update(self, v: float) -> None:
        pass

    def avg(self) -> float:
        raise

class Accuracy:
    def __init__(self) -> None:
        super().__init__()
        self.correct: int
        self.total: int

    def update(self, pred: Tensor, target: Tensor) -> None:
        pass

    def avg(self) -> float:
        raise