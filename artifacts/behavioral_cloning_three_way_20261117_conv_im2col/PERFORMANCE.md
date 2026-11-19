# CUDA im2col performance note

## Change

For float32 convolutions with dilation 1, the CUDA C backend now uses the
`im2col_f32` kernel to prepare the input patches used by the convolution
weight-gradient calculation. This avoids the CuPy path's separate padding,
temporary array, and repeated patch-slice assignments. Weight-gradient
contraction still uses CuPy, and Adam was not changed.

## Measurement

- GPU: NVIDIA GeForce RTX 5060
- Dataset: `car_behavioral_cloning_data`, 256 selected log rows / 768 images
- Model input: 3 x 32 x 64; batch size 8
- Timing: 24 training batches per repeat, 3 repeats; includes input transfer and
  synchronization, excludes image decoding
- CuPy backend: 129.47 ms per batch on average
- CUDA C backend: 107.52 ms per batch on average
- Difference: CUDA C was about 17.0% faster than CuPy in this run

The numerical check passed. Initial prediction max absolute error against the
NumPy reference was 2.09e-7; first-batch gradient relative L2 error was
5.06e-6. The full machine-readable results are in `benchmark.json`, with
per-step timings in `steps.csv` and the comparison plot in `benchmark.png`.

These numbers describe this model, dataset sample, GPU, and run; other shapes
and devices may see different results.
