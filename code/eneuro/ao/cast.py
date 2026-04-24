from __future__ import annotations
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import numpy as np

from ..base.core import Tensor, Function, Config
from ..nn.optim import Optimizer

@contextmanager
def autocast_context(dtype = np.float16):
    """
    上下文管理器：在此上下文中自动启用混合精度。
    用法：
        with autocast_context(np.float16):
            trainer.fit()
    """
    old_autocast_flag = Config.autocast
    old_dtype = Config.current_dtype
    
    Config.autocast = True
    Config.current_dtype = dtype
    try:
        yield
    finally:
        Config.autocast = old_autocast_flag
        Config.current_dtype = old_dtype

class GradScaler:
    '''
    动态损失缩放器，使梯度落入FP16的可表示范围内。
    用例：
    scaler = GradScaler()  # 损失缩放器
    for input, target in data:
        with autocast(dtype=torch.float16):   # 前向：低精度
            output = model(input)
            loss = loss_fn(output, target)

        scaler.scale(loss).backward()         # 反向：缩放后的低精度梯度计算
        scaler.step(optimizer)               # 反缩放梯度并更新参数
    '''
    def __init__(self, init_scale=32768.0, growth_factor=2.0, backoff_factor=0.5, 
                 growth_interval=2000, min_scale=1.0) -> None:
        self.scale_factor = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.min_scale = min_scale

        self.step_count = 0

    # 缩放增大梯度，防止下溢
    def scale(self, loss: Tensor) -> Tensor:
        return self.scale_factor * loss

    def step(self, optimizer: Optimizer) -> None:
        # 检查溢出
        has_overflow = False
        for param in optimizer.params:
            if param.grad is None:
                continue
            if (np.isinf(param.grad.data) | np.isnan(param.grad.data)).any():
                has_overflow = True
                break

        # 溢出则本次梯度无效
        if has_overflow:
            optimizer.zero_grad() # 清除梯度

            self.scale_factor *= self.backoff_factor # 缩小缩放倍数
            self.scale_factor = max(self.scale_factor, self.min_scale) # 不小于min_scale

            self.step_count = 0

        # 未溢出则梯度下降
        else:
            # 还原梯度大小，正常更新
            for param in optimizer.params:
                if param.grad is None:
                    continue
                param.grad.data /= self.scale_factor
            optimizer.step()

            # 连续未溢出则放大缩放倍数
            self.step_count += 1
            if self.step_count >= self.growth_interval:
                self.scale_factor *= self.growth_factor
                self.step_count = 0
        