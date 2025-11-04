from typing import Any

class Dataset:
    def __init__(self) -> None:
        pass

    def __getitem__(self) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError