"""Single-machine synchronous training with PS and Ring Allreduce backends."""

from .launcher import DistributedConfig, launch

__all__ = ['DistributedConfig', 'launch']
