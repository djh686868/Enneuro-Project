from .core import Tensor

class Parameter(Tensor):
    # when initialzing, data and name must be provided to superial class
    def __init__(self, data=None, name=None) -> None:
        super().__init__(data, name)
        self.requires_grad = True

    def backward(self, gradient: Tensor | None = None) -> None:
        return super().backward(gradient)
