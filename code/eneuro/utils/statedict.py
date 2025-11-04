class StateDict:
    def to_dict(self) -> dict:
        raise NotImplementedError
    
    def from_dict(self, d: dict) -> None:
        raise NotImplementedError
