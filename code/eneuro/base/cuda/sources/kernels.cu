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

__device__ __forceinline__ float winograd_load(
    const float* x, int sample, int channel, int ih, int iw,
    int channels, int height, int width) {
    return (ih >= 0 && ih < height && iw >= 0 && iw < width)
        ? x[((static_cast<int64_t>(sample) * channels + channel) * height + ih) * width + iw]
        : 0.0f;
}

// Winograd F(2x2, 3x3) forward convolution.  One thread computes one output
// element.  The four-by-four input tile and three-by-three filter are
// transformed with B^T d B and G g G^T, multiplied elementwise in the sixteen
// Winograd channels, and reconstructed with A^T M A.  The kernel is used for
// the common ResNet case stride=1, pad=1; edge tiles are explicitly masked.
extern "C" __global__ void winograd_f32(
    const float* x, const float* w, float* y,
    int n, int channels, int height, int width,
    int out_channels, int out_h, int out_w, int tiles_w) {
    const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(n) * out_channels * out_h * out_w;
    if (linear >= total) return;

    int64_t q = linear;
    const int ow = static_cast<int>(q % out_w); q /= out_w;
    const int oh = static_cast<int>(q % out_h); q /= out_h;
    const int oc = static_cast<int>(q % out_channels);
    const int sample = static_cast<int>(q / out_channels);
    const int tile_y = oh / 2;
    const int tile_x = ow / 2;
    const int local_y = oh & 1;
    const int local_x = ow & 1;

    float m00=0.0f,m01=0.0f,m02=0.0f,m03=0.0f;
    float m10=0.0f,m11=0.0f,m12=0.0f,m13=0.0f;
    float m20=0.0f,m21=0.0f,m22=0.0f,m23=0.0f;
    float m30=0.0f,m31=0.0f,m32=0.0f,m33=0.0f;

    for (int ic = 0; ic < channels; ++ic) {
        const int ih0=tile_y*2-1, ih1=ih0+1, ih2=ih0+2, ih3=ih0+3;
        const int iw0=tile_x*2-1, iw1=iw0+1, iw2=iw0+2, iw3=iw0+3;
        const float d00=winograd_load(x,sample,ic,ih0,iw0,channels,height,width), d01=winograd_load(x,sample,ic,ih0,iw1,channels,height,width), d02=winograd_load(x,sample,ic,ih0,iw2,channels,height,width), d03=winograd_load(x,sample,ic,ih0,iw3,channels,height,width);
        const float d10=winograd_load(x,sample,ic,ih1,iw0,channels,height,width), d11=winograd_load(x,sample,ic,ih1,iw1,channels,height,width), d12=winograd_load(x,sample,ic,ih1,iw2,channels,height,width), d13=winograd_load(x,sample,ic,ih1,iw3,channels,height,width);
        const float d20=winograd_load(x,sample,ic,ih2,iw0,channels,height,width), d21=winograd_load(x,sample,ic,ih2,iw1,channels,height,width), d22=winograd_load(x,sample,ic,ih2,iw2,channels,height,width), d23=winograd_load(x,sample,ic,ih2,iw3,channels,height,width);
        const float d30=winograd_load(x,sample,ic,ih3,iw0,channels,height,width), d31=winograd_load(x,sample,ic,ih3,iw1,channels,height,width), d32=winograd_load(x,sample,ic,ih3,iw2,channels,height,width), d33=winograd_load(x,sample,ic,ih3,iw3,channels,height,width);
        const float t00=d00-d20,t01=d01-d21,t02=d02-d22,t03=d03-d23;
        const float t10=d10+d20,t11=d11+d21,t12=d12+d22,t13=d13+d23;
        const float t20=d20-d10,t21=d21-d11,t22=d22-d12,t23=d23-d13;
        const float t30=d10-d30,t31=d11-d31,t32=d12-d32,t33=d13-d33;
        const float v00=t00-t02,v01=t01+t02,v02=t02-t01,v03=t01-t03;
        const float v10=t10-t12,v11=t11+t12,v12=t12-t11,v13=t11-t13;
        const float v20=t20-t22,v21=t21+t22,v22=t22-t21,v23=t21-t23;
        const float v30=t30-t32,v31=t31+t32,v32=t32-t31,v33=t31-t33;

        const int64_t wbase = (static_cast<int64_t>(oc) * channels + ic) * 9;
        const float g00=w[wbase],g01=w[wbase+1],g02=w[wbase+2],g10=w[wbase+3],g11=w[wbase+4],g12=w[wbase+5],g20=w[wbase+6],g21=w[wbase+7],g22=w[wbase+8];
        const float u00=g00,u01=.5f*(g00+g01+g02),u02=.5f*(g00-g01+g02),u03=g02;
        const float u10=g10,u11=.5f*(g10+g11+g12),u12=.5f*(g10-g11+g12),u13=g12;
        const float u20=g20,u21=.5f*(g20+g21+g22),u22=.5f*(g20-g21+g22),u23=g22;
        const float u30=g20,u31=.5f*(g20+g21+g22),u32=.5f*(g20-g21+g22),u33=g22;
        // Rows 1 and 2 of G are the half-sum/half-difference of filter rows;
        // the four columns below apply the same transform horizontally.
        const float q10=.5f*(g00+g10+g20),q11=.25f*(g00+g01+g02+g10+g11+g12+g20+g21+g22),q12=.25f*(g00-g01+g02+g10-g11+g12+g20-g21+g22),q13=.5f*(g02+g12+g22);
        const float q20=.5f*(g00-g10+g20),q21=.25f*(g00+g01+g02-g10-g11-g12+g20+g21+g22),q22=.25f*(g00-g01+g02-g10+g11-g12+g20-g21+g22),q23=.5f*(g02-g12+g22);
        m00+=v00*u00; m01+=v01*u01; m02+=v02*u02; m03+=v03*u03;
        m10+=v10*q10; m11+=v11*q11; m12+=v12*q12; m13+=v13*q13;
        m20+=v20*q20; m21+=v21*q21; m22+=v22*q22; m23+=v23*q23;
        m30+=v30*u30; m31+=v31*u31; m32+=v32*u32; m33+=v33*u33;
    }

    const float a00 = (m00 + m01 + m02) + (m10 + m11 + m12) + (m20 + m21 + m22);
    const float a01 = (m01 - m02 - m03) + (m11 - m12 - m13) + (m21 - m22 - m23);
    const float a10 = (m10 + m11 + m12) - (m20 + m21 + m22) - (m30 + m31 + m32);
    const float a11 = (m11 - m12 - m13) - (m21 - m22 - m23) - (m31 - m32 - m33);
    const float value = (local_y == 0 ? (local_x == 0 ? a00 : a01) : (local_x == 0 ? a10 : a11));
    y[((static_cast<int64_t>(sample) * out_channels + oc) * out_h + oh) * out_w + ow] = value;
}

// BatchNorm and global-average-pooling elementwise kernels. Reductions remain
// CuPy reductions so the existing dtype and stream semantics are preserved.
extern "C" __global__ void batchnorm_fwd_f32(
    const float* x, const float* mean, const float* var,
    const float* gamma, const float* beta, float* y,
    int64_t n, int64_t channels, int64_t spatial, float eps) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = n * channels * spatial;
    if (i < total) {
        const int64_t c = (i / spatial) % channels;
        const float inv = rsqrtf(var[c] + eps);
        y[i] = gamma[c] * (x[i] - mean[c]) * inv + beta[c];
    }
}

extern "C" __global__ void batchnorm_bwd_f32(
    const float* x, const float* gy, const float* mean, const float* inv_std,
    const float* gamma, const float* sum_gy, const float* sum_gy_xhat,
    float* gx, int64_t n, int64_t channels, int64_t spatial) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = n * channels * spatial;
    if (i < total) {
        const int64_t c = (i / spatial) % channels;
        const float xhat = (x[i] - mean[c]) * inv_std[c];
        const float m = static_cast<float>(n * spatial);
        gx[i] = gamma[c] * inv_std[c] * (m * gy[i] - sum_gy[c] - xhat * sum_gy_xhat[c]) / m;
    }
}

extern "C" __global__ void global_avg_fwd_f32(
    const float* x, float* y, int64_t n, int64_t channels, int64_t spatial) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = n * channels;
    if (i < total) {
        float sum = 0.0f;
        const int64_t base = i * spatial;
        for (int64_t k = 0; k < spatial; ++k) sum += x[base + k];
        y[i] = sum / static_cast<float>(spatial);
    }
}

extern "C" __global__ void global_avg_bwd_f32(
    const float* gy, float* gx, int64_t n, int64_t channels, int64_t spatial) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = n * channels * spatial;
    if (i < total) gx[i] = gy[i / spatial] / static_cast<float>(spatial);
}
