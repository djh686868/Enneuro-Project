__all__ = []

from .statedict import StateDict
__all__.append('StateDict')

__all__.extend(['Serializer', 'Visualizer', 'save_checkpoint', 'load_checkpoint'])

_HOOK_EXPORTS = (
    'HookHandle',
    'HookManager',
    'capture_features',
    'capture_gradients',
    'add_hooks_to_module'
)
__all__.extend(_HOOK_EXPORTS)


def __getattr__(name):
    if name in {'Serializer', 'save_checkpoint', 'load_checkpoint'}:
        from .serializer import Serializer

        return Serializer if name == 'Serializer' else getattr(Serializer, name)
    if name == 'Visualizer':
        from .visualization import Visualizer

        return Visualizer
    if name in _HOOK_EXPORTS:
        from . import hooks

        return getattr(hooks, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
