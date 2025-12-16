__all__ = []

from . import functions
__all__.append('functions')
from .functions import linear,conv2d,deconv2d,sigmoid,relu,softmax,pooling
__all__.append('linear')
__all__.append('conv2d')
__all__.append('deconv2d')
__all__.append('sigmoid')
__all__.append('relu')
__all__.append('softmax')
__all__.append('pooling')


from .core import Tensor, setup_tensor, as_Tensor, as_array, Config
__all__.append('Tensor')
setup_tensor()

from .parameter import Parameter
__all__.append('Parameter')