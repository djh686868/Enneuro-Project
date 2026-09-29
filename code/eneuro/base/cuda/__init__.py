"""Optional CUDA C (NVRTC/RawModule) backend.

This module is safe to import when CuPy or CUDA is unavailable.  The backend
is opt-in through ``ENNEURO_CUDA_BACKEND`` (default: ``cupy``).
"""
from .dispatch import *

