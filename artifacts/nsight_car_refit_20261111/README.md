# Car ResNet18 重训的 Nsight Systems 记录

本目录保存 2026-11-11 的 Car ResNet18 CUDA 重训采样。采样在 16 个预热批次后开始，覆盖随后 24 个训练批次。原始跟踪为 `training_batches.nsys-rep`，同名 SQLite 文件和各 CSV 为导出结果。

继续采集时，可开启 `ProfileConfig(nvtx=True, batch_every=1)`，由 Trainer 为各训练阶段生成 NVTX 标记。自定义循环可通过 `cuda_profiler_capture()` 在预热后采集指定批次。CLI 命令和报告查看方式见[训练性能采集指南](../../doc/training_profiling.md#nsight-systems-与-compute)。
