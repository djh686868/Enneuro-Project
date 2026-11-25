from importlib import import_module

__all__ = ["utils", "base", "nn", "data", "train", "ao", "explainability", "profiling"]
__version__ = "0.1dev"


def __getattr__(name):
    if name in __all__:
        return import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
