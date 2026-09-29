// Host-side launcher for the EnNeuro CUDA operator DLL.
//
// It includes kernels.cu so the DLL and RawModule compile exactly the same
// device formulas.  This layer never allocates or frees CuPy memory and never
// synchronizes; it only translates the stable C ABI into CUDA kernel launches.

#include "api.h"

#include <cuda_runtime.h>
#include <limits.h>
#include <stdint.h>

#include "kernels.cu"

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int64_t kMaxGridX = 2147483647LL;

enum Status : int32_t {
    kSuccess = 0,
    kInvalidArgument = 1,
    kUnsupported = 2,
    kCudaLaunchFailure = 3,
};

enum Operation : int32_t {
    kNeg = 0,
    kExp = 1,
    kLog = 2,
    kRelu = 3,
    kSigmoid = 4,
    kPow = 5,
    kAdd = 10,
    kSub = 11,
    kMul = 12,
    kDiv = 13,
    kReluBackward = 14,
    kBiasAdd = 20,
    kIm2Col = 30,
    // 31 (col2im) is deliberately unsupported in this stage: the validated
    // route uses a direct Conv2d input-gradient gather instead.
    kConv2dBackwardInput = 32,
    kMaxPoolForward = 40,
    kMaxPoolBackward = 41,
};

inline cudaStream_t as_cuda_stream(void* stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

inline bool valid_count(const int64_t count) {
    return count >= 0 && count <= kMaxGridX * static_cast<int64_t>(kThreadsPerBlock);
}

inline dim3 blocks_for(const int64_t count) {
    return dim3(static_cast<unsigned int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock));
}

inline int32_t launch_status() {
    // Do not synchronize: CUDA errors caused during execution are reported at
    // the caller's explicit stream synchronization boundary.  This only tests
    // configuration/immediate launch errors and preserves asynchronous timing.
    return cudaPeekAtLastError() == cudaSuccess ? kSuccess : kCudaLaunchFailure;
}

inline bool has_meta(const int64_t* meta, const int32_t meta_len, const int32_t needed) {
    return meta != nullptr && meta_len >= needed;
}

inline bool read_i32(const int64_t value, int* output) {
    if (value < INT_MIN || value > INT_MAX) return false;
    *output = static_cast<int>(value);
    return true;
}

inline bool read_dimensions(const int64_t* meta, const int32_t count, int* dimensions) {
    for (int32_t i = 0; i < count; ++i) {
        if (!read_i32(meta[i], &dimensions[i])) return false;
    }
    return true;
}

inline bool product(const int* values, const int count, int64_t* output) {
    int64_t value = 1;
    for (int i = 0; i < count; ++i) {
        if (values[i] < 0) return false;
        if (values[i] != 0 && value > INT64_MAX / values[i]) return false;
        value *= values[i];
    }
    *output = value;
    return valid_count(value);
}

inline bool valid_im2col_geometry(const int* d) {
    // d = N,C,H,W,KH,KW,OH,OW,SH,SW,PH,PW,DH,DW
    return d[0] >= 0 && d[1] >= 0 && d[2] >= 0 && d[3] >= 0 &&
           d[4] > 0 && d[5] > 0 && d[6] > 0 && d[7] > 0 &&
           d[8] > 0 && d[9] > 0 && d[10] >= 0 && d[11] >= 0 &&
           d[12] > 0 && d[13] > 0;
}

inline bool valid_pool_geometry(const int* d) {
    // d = N,C,H,W,KH,KW,OH,OW,SH,SW,PH,PW
    return d[0] >= 0 && d[1] >= 0 && d[2] >= 0 && d[3] >= 0 &&
           d[4] > 0 && d[5] > 0 && d[6] > 0 && d[7] > 0 &&
           d[8] > 0 && d[9] > 0 && d[10] >= 0 && d[11] >= 0;
}

#define ENEURO_LAUNCH_1D(kernel, count, stream, ...) \
    do { \
        if ((count) == 0) return kSuccess; \
        kernel<<<blocks_for(count), kThreadsPerBlock, 0, stream>>>(__VA_ARGS__); \
        return launch_status(); \
    } while (false)

}  // namespace

extern "C" ENEURO_CUDA_API int32_t eneuro_abi_version(void) {
    return 1;
}

extern "C" ENEURO_CUDA_API const char* eneuro_error_string(const int32_t status) {
    switch (status) {
        case kSuccess: return "success";
        case kInvalidArgument: return "invalid argument";
        case kUnsupported: return "unsupported ABI version, operation, or kernel feature";
        case kCudaLaunchFailure: return "CUDA kernel launch failed";
        default: return "unknown EnNeuro CUDA status";
    }
}

extern "C" ENEURO_CUDA_API int32_t eneuro_launch_v1(
    const int32_t op,
    const void* a,
    const void* b,
    void* out,
    void* aux,
    const int64_t* meta,
    const int32_t meta_len,
    const float scalar,
    void* stream) {
    if (meta_len < 0 || out == nullptr) return kInvalidArgument;
    if (meta_len > 0 && meta == nullptr) return kInvalidArgument;
    const cudaStream_t cuda_stream = as_cuda_stream(stream);

    switch (op) {
        case kNeg:
        case kExp:
        case kLog:
        case kRelu:
        case kSigmoid: {
            if (!has_meta(meta, meta_len, 1) || a == nullptr || !valid_count(meta[0])) return kInvalidArgument;
            const int64_t count = meta[0];
            if (op == kNeg) ENEURO_LAUNCH_1D(neg_f32, count, cuda_stream, static_cast<const float*>(a), static_cast<float*>(out), count);
            if (op == kExp) ENEURO_LAUNCH_1D(exp_f32, count, cuda_stream, static_cast<const float*>(a), static_cast<float*>(out), count);
            if (op == kLog) ENEURO_LAUNCH_1D(log_f32, count, cuda_stream, static_cast<const float*>(a), static_cast<float*>(out), count);
            if (op == kRelu) ENEURO_LAUNCH_1D(relu_f32, count, cuda_stream, static_cast<const float*>(a), static_cast<float*>(out), count);
            ENEURO_LAUNCH_1D(sigmoid_f32, count, cuda_stream, static_cast<const float*>(a), static_cast<float*>(out), count);
        }

        case kPow: {
            // Fast-route Pow validates a host scalar exponent.  Device-side
            // exponents are intentionally rejected until they have matching
            // tests, rather than silently using a different formula.
            if (!has_meta(meta, meta_len, 1) || a == nullptr || !valid_count(meta[0])) return kInvalidArgument;
            if (meta_len >= 2 && meta[1] != 0) return kUnsupported;
            const int64_t count = meta[0];
            ENEURO_LAUNCH_1D(pow_f32, count, cuda_stream, static_cast<const float*>(a), static_cast<float*>(out), scalar, count);
        }

        case kAdd:
        case kSub:
        case kMul:
        case kDiv: {
            // Both operands must be contiguous equal-shape buffers.  General
            // broadcasting remains a CuPy fallback in the current stage.
            if (!has_meta(meta, meta_len, 1) || a == nullptr || b == nullptr || !valid_count(meta[0])) return kInvalidArgument;
            const int64_t count = meta[0];
            if (op == kAdd) ENEURO_LAUNCH_1D(add_f32, count, cuda_stream, static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(out), count);
            if (op == kSub) ENEURO_LAUNCH_1D(sub_f32, count, cuda_stream, static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(out), count);
            if (op == kMul) ENEURO_LAUNCH_1D(mul_f32, count, cuda_stream, static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(out), count);
            ENEURO_LAUNCH_1D(div_f32, count, cuda_stream, static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(out), count);
        }

        case kReluBackward: {
            if (!has_meta(meta, meta_len, 1) || a == nullptr || b == nullptr || !valid_count(meta[0])) return kInvalidArgument;
            const int64_t count = meta[0];
            ENEURO_LAUNCH_1D(relu_bwd_f32, count, cuda_stream, static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(out), count);
        }

        case kBiasAdd: {
            if (!has_meta(meta, meta_len, 3) || a == nullptr || b == nullptr) return kInvalidArgument;
            const int64_t n = meta[0];
            const int64_t channels = meta[1];
            const int64_t spatial = meta[2];
            if (n < 0 || channels <= 0 || spatial <= 0 ||
                n > INT64_MAX / channels || n * channels > INT64_MAX / spatial) return kInvalidArgument;
            const int64_t count = n * channels * spatial;
            if (!valid_count(count)) return kInvalidArgument;
            ENEURO_LAUNCH_1D(bias_f32, count, cuda_stream, static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(out), n, channels, spatial);
        }

        case kIm2Col: {
            if (!has_meta(meta, meta_len, 14) || a == nullptr) return kInvalidArgument;
            int d[14];
            if (!read_dimensions(meta, 14, d) || !valid_im2col_geometry(d)) return kInvalidArgument;
            const int output_dims[] = {d[0], d[6], d[7], d[1], d[4], d[5]};
            int64_t count = 0;
            if (!product(output_dims, 6, &count)) return kInvalidArgument;
            ENEURO_LAUNCH_1D(im2col_f32, count, cuda_stream,
                static_cast<const float*>(a), static_cast<float*>(out),
                d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
                d[8], d[9], d[10], d[11], d[12], d[13]);
        }

        case kConv2dBackwardInput: {
            // meta = im2col meta plus OC.  kernels.cu has only dilation=1
            // gather, which is the path verified by the current test suite.
            if (!has_meta(meta, meta_len, 15) || a == nullptr || b == nullptr) return kInvalidArgument;
            int d[15];
            if (!read_dimensions(meta, 15, d) || !valid_im2col_geometry(d) || d[14] <= 0) return kInvalidArgument;
            if (d[12] != 1 || d[13] != 1) return kUnsupported;
            const int input_dims[] = {d[0], d[1], d[2], d[3]};
            int64_t count = 0;
            if (!product(input_dims, 4, &count)) return kInvalidArgument;
            ENEURO_LAUNCH_1D(conv_bwd_x_f32, count, cuda_stream,
                static_cast<const float*>(a), static_cast<const float*>(b), static_cast<float*>(out),
                d[0], d[1], d[2], d[3], d[14], d[4], d[5], d[6], d[7],
                d[8], d[9], d[10], d[11]);
        }

        case kMaxPoolForward: {
            if (!has_meta(meta, meta_len, 12) || a == nullptr || aux == nullptr) return kInvalidArgument;
            int d[12];
            if (!read_dimensions(meta, 12, d) || !valid_pool_geometry(d)) return kInvalidArgument;
            const int output_dims[] = {d[0], d[1], d[6], d[7]};
            int64_t count = 0;
            if (!product(output_dims, 4, &count)) return kInvalidArgument;
            ENEURO_LAUNCH_1D(pool_f32, count, cuda_stream,
                static_cast<const float*>(a), static_cast<float*>(out), static_cast<int64_t*>(aux),
                d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11]);
        }

        case kMaxPoolBackward: {
            if (!has_meta(meta, meta_len, 12) || a == nullptr || b == nullptr) return kInvalidArgument;
            int d[12];
            if (!read_dimensions(meta, 12, d) || !valid_pool_geometry(d)) return kInvalidArgument;
            const int output_dims[] = {d[0], d[1], d[6], d[7]};
            const int input_dims[] = {d[0], d[1], d[2], d[3]};
            int64_t count = 0;
            int64_t input_count = 0;
            if (!product(output_dims, 4, &count) || !product(input_dims, 4, &input_count)) return kInvalidArgument;
            if (input_count == 0) return kSuccess;
            // pool_bwd_f32 scatter-adds, so make its output deterministic from
            // the C ABI even if the Python wrapper supplied an empty buffer.
            if (cudaMemsetAsync(out, 0, static_cast<size_t>(input_count) * sizeof(float), cuda_stream) != cudaSuccess) return kCudaLaunchFailure;
            ENEURO_LAUNCH_1D(pool_bwd_f32, count, cuda_stream,
                static_cast<const float*>(a), static_cast<const int64_t*>(b), static_cast<float*>(out),
                d[0], d[1], d[2], d[3], d[6], d[7], d[4], d[5], d[8], d[9], d[10], d[11]);
        }

        default:
            return kUnsupported;
    }
}

#undef ENEURO_LAUNCH_1D
