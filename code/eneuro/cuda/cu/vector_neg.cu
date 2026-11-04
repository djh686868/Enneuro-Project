// vector_neg.cu  (逐元素取负: c = -a)
#include <cuda_runtime.h>
#include <stdio.h>

#if defined(_WIN32)
#define ENE_EXPORT extern "C" __declspec(dllexport)
#else
#define ENE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

__global__ void neg_kernel(const float *a, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = -a[idx];
    }
}

ENE_EXPORT
void launch_neg(const float *a, float *c, int n) {
    float *d_a = NULL, *d_c = NULL;
    size_t bytes = (size_t)n * sizeof(float);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    neg_kernel<<<blocks, threads>>>(d_a, d_c, n);

    cudaMemcpy(c, d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_c);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// ---------------------------------------------------------------------------
// 方案B：零拷贝接口（只借设备指针，不分配不拷贝；stream 由调用方传入）
// ---------------------------------------------------------------------------
ENE_EXPORT
void launch_neg_dev(const float *d_a, float *d_c, int n, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    neg_kernel<<<blocks, threads, 0, s>>>(d_a, d_c, n);
}
