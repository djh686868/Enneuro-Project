"""Validate and average per-worker mean gradients."""

import numpy as np


def aggregate(contributions, parameter_shapes):
    if not contributions:
        raise ValueError('no gradients received')
    total = sum(count for count, _ in contributions)
    if total <= 0:
        raise ValueError('empty batch')
    expected = set(parameter_shapes)
    result = {}
    for count, gradients in contributions:
        if count <= 0 or set(gradients) != expected:
            raise ValueError('gradient names or sample count mismatch')
        for name, shape in parameter_shapes.items():
            value = np.asarray(gradients[name])
            if value.shape != tuple(shape) or not np.all(np.isfinite(value)):
                raise ValueError(f'invalid gradient for {name}')
            if name not in result:
                result[name] = np.zeros(shape, dtype=np.result_type(value.dtype, np.float64))
            result[name] += value * count
    return {name: value / total for name, value in result.items()}
