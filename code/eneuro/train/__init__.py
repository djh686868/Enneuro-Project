__all__ = []

from .meters import AverageMeter
__all__.append('AverageMeter')

from .trainer import Trainer, Evaluator
__all__.append('Trainer')
__all__.append('Evaluator')

from .metrics import accuracy, confusion_matrix, precision_recall_f1, ClassificationReport
__all__.append('accuracy')
__all__.append('confusion_matrix')
__all__.append('precision_recall_f1')
__all__.append('ClassificationReport')

# Import the distributed runtime on demand. Its workers use Trainer too.
__all__.extend(['DistributedConfig', 'launch_distributed'])


def __getattr__(name):
    if name == 'DistributedConfig':
        from ..distributed import DistributedConfig
        return DistributedConfig
    if name == 'launch_distributed':
        from ..distributed import launch
        return launch
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
