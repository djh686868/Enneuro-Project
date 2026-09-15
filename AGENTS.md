# EnNeuro 项目说明

## CUDA 版本兼容性

当运行测试或代码时遇到因 **CUDA 版本过新导致 API 不支持** 的错误（例如 `cupy` 或 CUDA kernel 相关的 `CUDADriverError`、`cupy.cuda.driver.CUDADriverError` 等），请临时将 CUDA 切换到 **12.6** 版本。

机器上已安装 CUDA 12.6，路径为：

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
```

**切换方式（在当前终端会话中临时生效）：**

PowerShell：
```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;" + $env:PATH
```

CMD：
```cmd
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%
```

切换后重新运行失败的命令即可。此操作仅对当前终端会话有效，不影响系统全局配置。
