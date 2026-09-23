// EnNeuro stage-one CUDA C device kernels.
//
// This is the single device-source file for both routes:
//   * RawModule/NVRTC reads and compiles this file at runtime;
//   * library.cu includes this file and nvcc compiles it into the DLL.
// Do not add host launch wrappers here, otherwise NVRTC and DLL would have
// separate formula implementations.

// NVRTC used by CuPy does not ship a host-style stdint.h include path on this
// Windows setup.  Give its runtime compiler the same explicit 64-bit type;
// nvcc builds use the normal C header.  Both branches therefore have exactly
// the same ABI width for tensor offsets and pool indexes.
#ifdef __CUDACC_RTC__
typedef long long int64_t;
#else
#include <stdint.h>
#endif

// Every simple kernel uses one output element per thread.  The host launcher
// uses 256 threads/block, and all kernel indices are int64_t so Windows host
// `long` width never affects tensor indexing.

extern "C" __global__ void add_f32(const float* a, const float* b, float* y, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}

extern "C" __global__ void sub_f32(const float* a, const float* b, float* y, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] - b[i];
}

extern "C" __global__ void mul_f32(const float* a, const float* b, float* y, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] * b[i];
}

extern "C" __global__ void div_f32(const float* a, const float* b, float* y, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] / b[i];
}

extern "C" __global__ void neg_f32(const float* x, float* y, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = -x[i];
}

extern "C" __global__ void exp_f32(const float* x, float* y, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = expf(x[i]);
}

extern "C" __global__ void log_f32(const float* x, float* y, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = logf(x[i]);
}

extern "C" __global__ void pow_f32(const float* x, float* y, float exponent, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = powf(x[i], exponent);
}

extern "C" __global__ void relu_f32(const float* x, float* y, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

extern "C" __global__ void relu_bwd_f32(const float* x, const float* gy, float* gx, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) gx[i] = x[i] > 0.0f ? gy[i] : 0.0f;
}

extern "C" __global__ void sigmoid_f32(const float* x, float* y, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // Equivalent to 1/(1+exp(-x)), but avoids direct exp overflow.
    if (i < n) y[i] = 0.5f * tanhf(0.5f * x[i]) + 0.5f;
}

extern "C" __global__ void bias_f32(
    const float* x, const float* bias, float* y,
    int64_t n, int64_t channels, int64_t spatial) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = n * channels * spatial;
    if (i < total) {
        // spatial = 1 for NC, and H*W for contiguous NCHW.
        const int64_t channel = (i / spatial) % channels;
        y[i] = x[i] + bias[channel];
    }
}

extern "C" __global__ void im2col_f32(
    const float* x, float* col,
    int n, int channels, int height, int width,
    int kernel_h, int kernel_w, int out_h, int out_w,
    int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(n) * out_h * out_w * channels * kernel_h * kernel_w;
    if (t >= total) return;

    // The matrix has row (sample, oh, ow) and column (channel, kh, kw),
    // with kw fastest.  This exactly matches col.dot(W.reshape(OC,-1).T).
    const int kw = static_cast<int>(t % kernel_w); t /= kernel_w;
    const int kh = static_cast<int>(t % kernel_h); t /= kernel_h;
    const int channel = static_cast<int>(t % channels); t /= channels;
    const int ow = static_cast<int>(t % out_w); t /= out_w;
    const int oh = static_cast<int>(t % out_h); t /= out_h;
    const int sample = static_cast<int>(t);

    const int ih = oh * stride_h + kh * dilation_h - pad_h;
    const int iw = ow * stride_w + kw * dilation_w - pad_w;
    const int64_t col_index =
        (((static_cast<int64_t>(sample) * out_h + oh) * out_w + ow) * channels * kernel_h * kernel_w)
        + (static_cast<int64_t>(channel) * kernel_h * kernel_w + kh * kernel_w + kw);
    col[col_index] = (ih >= 0 && ih < height && iw >= 0 && iw < width)
        ? x[((static_cast<int64_t>(sample) * channels + channel) * height + ih) * width + iw]
        : 0.0f;
}

extern "C" __global__ void pool_f32(
    const float* x, float* y, int64_t* indexes,
    int n, int channels, int height, int width,
    int kernel_h, int kernel_w, int out_h, int out_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(n) * channels * out_h * out_w;
    if (t >= total) return;

    const int ow = static_cast<int>(t % out_w); t /= out_w;
    const int oh = static_cast<int>(t % out_h); t /= out_h;
    const int channel = static_cast<int>(t % channels);
    const int sample = static_cast<int>(t / channels);

    // Padding is zero to remain compatible with EnNeuro's im2col reference.
    // Strict `>` retains the first location when values tie, matching argmax.
    float best = 0.0f;
    int64_t best_local_index = 0;
    bool found = false;
    for (int kh = 0; kh < kernel_h; ++kh) {
        for (int kw = 0; kw < kernel_w; ++kw) {
            const int ih = oh * stride_h + kh - pad_h;
            const int iw = ow * stride_w + kw - pad_w;
            const float value = (ih >= 0 && ih < height && iw >= 0 && iw < width)
                ? x[((static_cast<int64_t>(sample) * channels + channel) * height + ih) * width + iw]
                : 0.0f;
            const int64_t local_index = static_cast<int64_t>(kh) * kernel_w + kw;
            if (!found || value > best) {
                best = value;
                best_local_index = local_index;
                found = true;
            }
        }
    }
    const int64_t output_index = ((static_cast<int64_t>(sample) * channels + channel) * out_h + oh) * out_w + ow;
    y[output_index] = best;
    indexes[output_index] = best_local_index;
}

extern "C" __global__ void pool_bwd_f32(
    const float* gy, const int64_t* indexes, float* gx,
    int n, int channels, int height, int width, int out_h, int out_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(n) * channels * out_h * out_w;
    if (t >= total) return;

    const int ow = static_cast<int>(t % out_w); t /= out_w;
    const int oh = static_cast<int>(t % out_h); t /= out_h;
    const int channel = static_cast<int>(t % channels);
    const int sample = static_cast<int>(t / channels);
    const int64_t output_index = ((static_cast<int64_t>(sample) * channels + channel) * out_h + oh) * out_w + ow;
    const int64_t local_index = indexes[output_index];
    const int kh = static_cast<int>(local_index / kernel_w);
    const int kw = static_cast<int>(local_index % kernel_w);
    const int ih = oh * stride_h + kh - pad_h;
    const int iw = ow * stride_w + kw - pad_w;

    if (ih >= 0 && ih < height && iw >= 0 && iw < width) {
        // Pooling windows can overlap.  atomicAdd gives the required sum of
        // all contributing output gradients; addition order can vary slightly.
        atomicAdd(&gx[((static_cast<int64_t>(sample) * channels + channel) * height + ih) * width + iw], gy[output_index]);
    }
}

extern "C" __global__ void conv_bwd_x_f32(
    const float* gy, const float* weight, float* gx,
    int n, int in_channels, int height, int width,
    int out_channels, int kernel_h, int kernel_w, int out_h, int out_w,
    int stride_h, int stride_w, int pad_h, int pad_w) {
    int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(n) * in_channels * height * width;
    if (t >= total) return;

    const int iw = static_cast<int>(t % width); t /= width;
    const int ih = static_cast<int>(t % height); t /= height;
    const int input_channel = static_cast<int>(t % in_channels);
    const int sample = static_cast<int>(t / in_channels);

    // One thread owns one gx element and gathers all eligible gy*W terms.
    // This eliminates a large col2im workspace and avoids atomic writes.
    float accumulated = 0.0f;
    for (int output_channel = 0; output_channel < out_channels; ++output_channel) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                const int oh_numerator = ih + pad_h - kh;
                const int ow_numerator = iw + pad_w - kw;
                if (oh_numerator < 0 || ow_numerator < 0 ||
                    oh_numerator % stride_h != 0 || ow_numerator % stride_w != 0) {
                    continue;
                }
                const int oh = oh_numerator / stride_h;
                const int ow = ow_numerator / stride_w;
                if (oh < out_h && ow < out_w) {
                    const int64_t gy_index = ((static_cast<int64_t>(sample) * out_channels + output_channel) * out_h + oh) * out_w + ow;
                    const int64_t weight_index =
                        ((static_cast<int64_t>(output_channel) * in_channels + input_channel) * kernel_h + kh) * kernel_w + kw;
                    accumulated += gy[gy_index] * weight[weight_index];
                }
            }
        }
    }
    gx[((static_cast<int64_t>(sample) * in_channels + input_channel) * height + ih) * width + iw] = accumulated;
}
