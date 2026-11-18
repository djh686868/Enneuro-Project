// Stable C ABI for the EnNeuro CUDA operator DLL.
// The ABI deliberately uses only raw device pointers and host-side metadata so
// that CuPy remains the owner of device allocations and CUDA streams.

#ifndef ENEURO_CUDA_API_H_
#define ENEURO_CUDA_API_H_

#include <stdint.h>

#if defined(_WIN32)
#  if defined(ENEURO_CUDA_BUILD)
#    define ENEURO_CUDA_API __declspec(dllexport)
#  else
#    define ENEURO_CUDA_API __declspec(dllimport)
#  endif
#else
#  define ENEURO_CUDA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ABI version expected by Python ctypes bindings.
ENEURO_CUDA_API int32_t eneuro_abi_version(void);

// Launch one already-validated float32 CUDA kernel.
//
// a, b, out and aux are CUDA device pointers.  meta points to a host int64
// array and is read before this function returns.  stream is a CUDA stream
// address (CuPy: cp.cuda.get_current_stream().ptr); NULL means the CUDA null
// stream.  The function does not synchronize the stream.
//
// Return statuses: 0 = success, 1 = invalid argument, 2 = ABI/op unsupported,
// 3 = immediate CUDA launch failure.
ENEURO_CUDA_API int32_t eneuro_launch_v1(
    int32_t op,
    const void* a,
    const void* b,
    void* out,
    void* aux,
    const int64_t* meta,
    int32_t meta_len,
    float scalar,
    void* stream);

// Maps the above status codes to static human-readable strings.
ENEURO_CUDA_API const char* eneuro_error_string(int32_t status);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ENEURO_CUDA_API_H_
