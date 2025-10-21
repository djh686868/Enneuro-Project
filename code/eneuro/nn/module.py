from ..core import Tensor, Parameter
from ..utils import StateDict

class Module(StateDict):
    params: list[Parameter]

    _modules: dict[str, "Module"] # 子层
    _params: dict[str, Parameter] # 直接参数
    
    def __init__(self) -> None:
        pass

    def __call__(self, x: Tensor) -> Tensor:
        raise

    def forward(self, x: Tensor) -> Tensor:
        raise

    def to_dict(self) -> dict: # 递归调用子层/参数
        return super().to_dict()
    
    def from_dict(self, d: dict) -> None: # 反向重建
        return super().from_dict(d)
    
class Linear(Module):
    weight : Parameter
    bias : Parameter | None
    
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
    
    def to_dict(self) -> dict:
        return super().to_dict()
    
    def from_dict(self, d: dict) -> None:
        return super().from_dict(d)
    
class Sequential(Module):
    layers: list[Module]

    def __init__(self, *layers: Module) -> None:
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)
    
    def to_dict(self) -> dict:
        return super().to_dict()
    
    def from_dict(self, d: dict) -> None:
        return super().from_dict(d)