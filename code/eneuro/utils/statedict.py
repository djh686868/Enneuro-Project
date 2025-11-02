

from abc import ABC, abstractmethod

class StateDict(ABC):
    """状态字典抽象基类，只定义接口"""
    
    @abstractmethod
    def to_dict(self):
        """将状态转换为字典"""
        pass
    
    @abstractmethod
    def from_dict(self, data):
        """从字典恢复状态"""
        pass
