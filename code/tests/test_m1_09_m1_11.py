# -*- coding: utf-8 -*-
"""Contract tests for the M1-09 benchmark and M1-11 CUDA C switch."""

import csv
import importlib.util
import sys
import unittest
import uuid
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from eneuro.base import Tensor  # noqa: E402
from eneuro.base.cuda import dispatch  # noqa: E402
from eneuro.base.functions import (  # noqa: E402
    conv2d,
    get_backend_capabilities,
    pooling,
)

BENCHMARK_PATH = Path(__file__).resolve().parent / 'bench_m1_09_tri_backend.py'
SPEC = importlib.util.spec_from_file_location(
    'bench_m1_09_tri_backend', BENCHMARK_PATH
)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def _cuda_c_ready():
    try:
        return bool(get_backend_capabilities()['cuda_c']['available'])
    except Exception:
        return False


class BackendContractTests(unittest.TestCase):
    def test_cuda_c_alias_and_extension_status(self):
        previous = dispatch.get_backend()
        try:
            dispatch.set_backend('cuda_c')
            self.assertEqual(dispatch.get_backend(), 'rawmodule')
            self.assertFalse(dispatch.is_available('extension'))
        finally:
            dispatch.set_backend(previous)

    def test_numpy_per_call_switch(self):
        rng = np.random.default_rng(7)
        x = Tensor(rng.normal(size=(1, 2, 5, 5)).astype(np.float32))
        weight = Tensor(rng.normal(size=(3, 2, 3, 3)).astype(np.float32))
        bias = Tensor(rng.normal(size=3).astype(np.float32))

        conv_result = conv2d(x, weight, bias, pad=1, backend='numpy')
        pool_result = pooling(x, kernel_size=2, stride=2, backend='numpy')
        self.assertEqual(conv_result.creator._backend_effective, 'numpy')
        self.assertEqual(pool_result.creator._backend_effective, 'numpy')

    @unittest.skipUnless(_cuda_c_ready(), 'CUDA C RawModule is unavailable')
    def test_conv_and_pool_execute_native_cuda_c(self):
        import cupy as cp

        rng = np.random.default_rng(17)
        x = Tensor(cp.asarray(
            rng.normal(size=(1, 2, 6, 6)).astype(np.float32)
        ), requires_grad=True, device='cuda')
        weight = Tensor(cp.asarray(
            rng.normal(size=(3, 2, 3, 3)).astype(np.float32)
        ), requires_grad=True, device='cuda')
        bias = Tensor(
            cp.asarray(rng.normal(size=3).astype(np.float32)),
            requires_grad=True, device='cuda',
        )

        conv_cupy = conv2d(x, weight, bias, pad=1, backend='cupy')
        conv_cuda = conv2d(x, weight, bias, pad=1, backend='cuda_c')
        conv_auto = conv2d(x, weight, bias, pad=1, backend='auto')
        pool_cupy = pooling(x, kernel_size=2, stride=2, backend='cupy')
        pool_cuda = pooling(x, kernel_size=2, stride=2, backend='cuda_c')
        pool_auto = pooling(x, kernel_size=2, stride=2, backend='auto')
        cp.cuda.Stream.null.synchronize()

        np.testing.assert_allclose(
            cp.asnumpy(conv_cuda.data), cp.asnumpy(conv_cupy.data),
            rtol=2e-4, atol=5e-5,
        )
        np.testing.assert_allclose(
            cp.asnumpy(pool_cuda.data), cp.asnumpy(pool_cupy.data),
            rtol=0, atol=0,
        )
        self.assertEqual(conv_cuda.creator._backend_effective, 'cuda_c')
        self.assertEqual(pool_cuda.creator._backend_effective, 'cuda_c')
        self.assertEqual(conv_auto.creator._backend_effective, 'cuda_c')
        self.assertEqual(pool_auto.creator._backend_effective, 'cuda_c')
        self.assertTrue(conv_cuda.creator._backend_launch_counts)
        self.assertGreater(
            pool_cuda.creator._backend_launch_counts.get('pool_f32', 0), 0
        )

        # The current CUDA C Conv2D kernel does not implement dilation.  Keep
        # the CuPy fallback visible instead of reporting it as native CUDA C.
        conv_dilated = conv2d(
            x, weight, bias, pad=2, dilation=2, backend='cuda_c'
        )
        self.assertEqual(
            conv_dilated.creator._backend_effective, 'cupy_fallback'
        )
        self.assertIn(
            'dilation', conv_dilated.creator._backend_fallback_reason
        )

        dispatch.launch_counts(reset=True)
        conv_grads = conv_cuda.creator.backward(Tensor(
            cp.ones_like(conv_cuda.data), device='cuda'
        ))
        pool_grad = pool_cuda.creator.backward(Tensor(
            cp.ones_like(pool_cuda.data), device='cuda'
        ))
        backward_counts = dispatch.launch_counts()
        self.assertGreater(backward_counts.get('conv_bwd_x_f32', 0), 0)
        self.assertGreater(backward_counts.get('pool_bwd_f32', 0), 0)
        self.assertEqual(conv_grads[0].shape, x.shape)
        self.assertEqual(conv_grads[1].shape, weight.shape)
        self.assertEqual(pool_grad.shape, x.shape)


class BenchmarkSmokeTests(unittest.TestCase):
    @unittest.skipUnless(_cuda_c_ready(), 'CUDA C RawModule is unavailable')
    def test_tiny_three_backend_run_records_native_cuda_c(self):
        artifact_root = Path(__file__).resolve().parents[2] / 'tests' / 'artifacts'
        token = uuid.uuid4().hex
        csv_path = artifact_root / f'.m1_09_test_{token}.csv'
        plot_path = artifact_root / f'.m1_09_test_{token}.png'
        metadata_path = artifact_root / f'.m1_09_test_{token}.json'
        try:
            result = benchmark.main([
                '--ops', 'conv', 'pool',
                '--backends', 'numpy', 'cupy', 'cuda_c',
                '--shapes', '1,2,6,6',
                '--kernels', '3,3,3',
                '--iters', '1',
                '--warmup', '0',
                '--out-csv', str(csv_path),
                '--out-plot', str(plot_path),
                '--out-meta', str(metadata_path),
                '--fail-on-error',
            ])
            self.assertEqual(result, 0)
            self.assertTrue(csv_path.exists())
            self.assertTrue(metadata_path.exists())
            with csv_path.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))

            self.assertEqual(len(rows), 6)
            cuda_rows = [row for row in rows if row['backend'] == 'cuda_c']
            self.assertEqual(len(cuda_rows), 2)
            for row in cuda_rows:
                self.assertEqual(row['status'], 'ok')
                self.assertEqual(row['effective_backend'], 'cuda_c')
                self.assertEqual(row['native_cuda_c'], 'True')
                self.assertEqual(row['timing_source'], 'cuda_event')
                self.assertNotEqual(row['kernel_launches'], '{}')
        finally:
            csv_path.unlink(missing_ok=True)
            plot_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
