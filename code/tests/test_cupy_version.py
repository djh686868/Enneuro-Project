import time
import cupy as cp
cp.show_config()

from cupy.cuda import cublas

# 1. 创建 cuBLAS handle
handle = cublas.create()

# 2. 获取版本号（返回整数，例如 120200 代表 12.2.0）
version = cublas.getVersion(handle)
print(f"cuBLAS version: {version}")

# 3. 销毁 handle（释放资源）
cublas.destroy(handle)

a = cp.random.random((128, 256))
b = cp.random.random((256, 128))

c = cp.dot(a, b)

cp.cuda.Stream.null.synchronize()
start = time.perf_counter()
for _ in range(100):
    c = cp.dot(a, b)
cp.cuda.Stream.null.synchronize()
print(f"用时: {time.perf_counter() - start:.4f} 秒")