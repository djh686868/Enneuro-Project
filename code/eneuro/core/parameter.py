from .tensor import Tensor

class Parameter(Tensor):
    def __init__(self) -> None:
        super().__init__()
        self.requires_grad = True

    def backward(self, gradient: Tensor | None = None) -> None:
        return super().backward(gradient)
