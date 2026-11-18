// vector_dot.cu  (归约点积: 返回 dot(a, b) = sum(a * b))
// 两阶段归约: 块内共享内存折半归约 -> 每个块只做一次 atomicAdd
#include <cuda_runtime.h>
#include <stdio.h>

#if defined(_WIN32)
#define ENE_EXPORT extern "C" __declspec(dllexport)
#else
#define ENE_EXPORT extern "C" __attribute__((visibility("default")))
#endif

__global__ void dot_kernel(const float *a, const float *b, float *out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

// 返回值即为点积结果
ENE_EXPORT
float launch_dot(const float *a, const float *b, int n) {
    float *d_a = NULL, *d_b = NULL, *d_out = NULL;
    size_t bytes = (size_t)n * sizeof(float);
    float result = 0.0f;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    dot_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_a, d_b, d_out, n);

    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    return result;
}

// ---------------------------------------------------------------------------
// 方案B：零拷贝接口（归约结果写入调用方常驻的 1 元素设备缓冲 d_out）
// ---------------------------------------------------------------------------
ENE_EXPORT
void launch_dot_dev(const float *d_a, const float *d_b, int n, float *d_out, void *stream) {
    cudaStream_t s = (cudaStream_t)stream;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    cudaMemsetAsync(d_out, 0, sizeof(float), s);
    dot_kernel<<<blocks, threads, threads * sizeof(float), s>>>(d_a, d_b, d_out, n);
}
