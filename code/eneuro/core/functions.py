from .tensor import Tensor

class Function:
    def __init__(self) -> None:
        pass

    def forward(self, *t : Tensor) -> None:
        pass

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return ()

class Add(Function):
    def __init__(self, a: Tensor, b: Tensor) -> None:
        self.a: Tensor = a
        self.b: Tensor = b

    def forward(self, *t : Tensor) -> None:
        pass

    def backward(self, grad_out: Tensor) -> tuple[Tensor, ...]:
        return ()
