// axpy.cu  (融合乘加, 原地更新: y = alpha * x + y)
#include <cuda_runtime.h>
#include <stdio.h>

#if defined(_WIN32)
#define ENE_EXPORT extern "C" __declspec(dllexport)
#else
#define ENE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

__global__ void axpy_kernel(const float *x, float *y, float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = alpha * x[idx] + y[idx];
    }
}

// 注意: y 为输入兼输出 (原地更新), 调用方需保证可写
ENE_EXPORT
void launch_axpy(const float *x, float *y, float alpha, int n) {
    float *d_x = NULL, *d_y = NULL;
    size_t bytes = (size_t)n * sizeof(float);

    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);

    cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    axpy_kernel<<<blocks, threads>>>(d_x, d_y, alpha, n);

    cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------------
// 方案B：零拷贝接口（只借设备指针，不分配不拷贝；stream 由调用方传入）
// 注意：y 为输入兼输出（原地更新），零拷贝时直接改 cupy 的显存，无回写成本
// ---------------------------------------------------------------------------
ENE_EXPORT
void launch_axpy_dev(const float *d_x, float *d_y, float alpha, int n, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    axpy_kernel<<<blocks, threads, 0, s>>>(d_x, d_y, alpha, n);
}
