"""Single-machine synchronous parameter-server training."""

from .launcher import DistributedConfig, launch

__all__ = ['DistributedConfig', 'launch']
