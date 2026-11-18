// vector_add.cu
#include <cuda_runtime.h>
#include <stdio.h>

#if defined(_WIN32)
#define ENE_EXPORT extern "C" __declspec(dllexport)
#else
#define ENE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

__global__ void add_kernel(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

ENE_EXPORT
void launch_add(float *a, float *b, float *c, int n) {
    float *d_a, *d_b, *d_c;
    size_t bytes = n * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------------
// 方案B：显存常驻 / 零拷贝接口
//   调用方自己持有设备显存（例如 cupy 的 ndarray），本函数只借用指针。
//   · 不分配、不拷贝，彻底消除 cudaMalloc/H2D/D2H 开销
//   · nvcc 的运行时 API 与 cupy 共用同一个 primary context，
//     故可直接传入 cupy 的 ndarray.data.ptr（要求同一张卡）
//   · stream 由调用方传入，保证与 cupy 的非默认流有序；传 0 即默认流
// ---------------------------------------------------------------------------
ENE_EXPORT
void launch_add_dev(const float *d_a, const float *d_b, float *d_c, int n, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, s>>>((float *)d_a, (float *)d_b, d_c, n);
}