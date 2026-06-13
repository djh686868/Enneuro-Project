"""
Grad-CAM (Gradient-weighted Class Activation Mapping) 实现

参考论文: "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
https://arxiv.org/abs/1610.02391

设计说明：
- 完全使用 eneuro.utils.hooks 模块来捕获特征图和梯度
- GradCAM 通过 capture_features 和 capture_gradients 使用钩子功能
"""

import numpy as np
from typing import Dict, Optional, List

from ..base import Tensor
from ..nn.module import Module, Layer
from ..base import functions as F
from ..utils import capture_features, capture_gradients
from ..utils.hooks import HookRegistry


class GradCAM:
    """
    Grad-CAM 类激活映射计算

    核心公式：
    α_k^c = (1/Z) Σ_i Σ_j ∂Y^c / ∂A_ij^k
    L_(grad-CAM)^c = ReLU(Σ_k α_k^c · A^k)

    其中：
    - A_ij^k: 第k个卷积核在位置(i,j)的激活值
    - Y^c: 目标类别c的logits得分
    - α_k^c: 类别c关于第k个特征图的神经元重要性权重
    """

    def __init__(self, model: Module, target_layer: Layer):
        """
        初始化GradCAM

        Args:
            model: 神经网络模型
            target_layer: 目标层（通常是最后一个卷积层）
        """
        self.model = model
        self.target_layer = target_layer

        self._feature_handle = None
        self._gradient_handle = None
        self._feature_storage = None
        self._gradient_storage = None

    def _remove_hooks(self):
        """移除所有已注册的钩子"""
        if self._feature_handle is not None:
            self._feature_handle.remove()
            self._feature_handle = None
            self._feature_storage = None

        if self._gradient_handle is not None:
            self._gradient_handle.remove()
            self._gradient_handle = None
            self._gradient_storage = None

    def generate(
        self,
        input_tensor: Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        生成Grad-CAM热力图

        实现说明
        --------
        当 target_layer 之后紧跟 BatchNorm 时，BN 的反向传播公式
        会使 ∂L/∂(conv_output) 的空间均值严格为 0（每通道），
        导致 alpha_k 全零、热力图全黑。

        修复方案：在前向传播中追踪层调用顺序，找到 target_layer
        的后继层（通常是 BN 层），在后继层的 backward hook 处捕获
        梯度 gy = ∂L/∂(BN_output)——此时梯度尚未被 BN backward
        处理，空间均值非零，alpha_k 有意义。

        特征图仍取自 target_layer 的前向输出（conv 输出），
        梯度取自后继层 backward 的 gy；两者空间分辨率相同，
        Grad-CAM 公式依然成立。

        Args:
            input_tensor: 输入张量 (N, C, H, W)
            class_idx: 目标类别索引，为 None 时使用预测最高置信度类别

        Returns:
            热力图 (H, W)，已归一化到 [0, 1]
        """
        try:
            # ── Step 1: 注册特征钩子，同时追踪层调用顺序 ──────────────────
            self._feature_handle, self._feature_storage = capture_features(self.target_layer)

            registry = HookRegistry()
            registry.start_recording_sequence()
            output = self.model(input_tensor)
            registry.stop_recording_sequence()

            # ── Step 2: 确定梯度捕获层 ─────────────────────────────────────
            # 优先使用后继层（绕开 BN backward 使梯度均值归零的问题），
            # 但要求后继层与目标层通道数一致（BatchNorm 通道数必定相同），
            # Conv→Conv 时若通道数不同则不使用后继层。
            act_channels = self._feature_storage['output'].shape[1]  # 目标层输出通道数

            gradient_layer = registry.get_successor_layer(self.target_layer)
            if gradient_layer is not None:
                successor_type = type(gradient_layer).__name__
                # BatchNorm 输入/输出通道数与前层相同 → 安全
                if 'BatchNorm' in successor_type:
                    pass  # 使用后继层
                else:
                    # 其他类型：只有通道数确实能对上才使用
                    succ_out = getattr(gradient_layer, 'out_channels',
                               getattr(gradient_layer, 'out_size', None))
                    if succ_out != act_channels:
                        gradient_layer = self.target_layer
            else:
                gradient_layer = self.target_layer

            # 梯度钩子可在前向之后注册：backward 时按层的 _hook_manager 查找，
            # 与前向时间无关，因此仍能被正确触发。
            self._gradient_handle, self._gradient_storage = capture_gradients(gradient_layer)

            # ── Step 3: 反向传播 ────────────────────────────────────────────
            if class_idx is None:
                class_idx = int(np.argmax(output.data, axis=1)[0])

            target_logit = output[0, class_idx]
            target_logit.backward()

            # ── Step 4: 取出特征图和梯度 ────────────────────────────────────
            activations = self._feature_storage['output']
            gradients = self._gradient_storage['grad_output']

            if activations is None:
                raise RuntimeError(
                    "未能捕获特征图。请确保目标层正确注册了钩子。"
                )
            if gradients is None:
                raise RuntimeError(
                    "未能捕获梯度。请确认后继层（BatchNorm/激活层）"
                    "支持梯度计算，或手动指定 gradient_layer。"
                )

            # ── Step 5: Grad-CAM 核心计算 ───────────────────────────────────
            # activations: (N, C, H, W) → 取第 0 个样本 → (C, H, W)
            # gradients:   (N, C, H, W) → 取第 0 个样本 → (C, H, W)
            activations = activations[0]   # (C, H, W)
            gradients   = gradients[0]     # (C, H, W) 或 (C,)

            # α_k = GlobalAveragePool(∂Y^c / ∂A^k) → shape (C,)
            if gradients.ndim == 3:
                alpha_k_arr = gradients.mean(axis=(1, 2))   # (C,)
            elif gradients.ndim == 2:
                alpha_k_arr = gradients.mean(axis=1)        # (C,)
            else:
                alpha_k_arr = gradients                     # already (C,)

            # L = ReLU(Σ_k α_k · A^k)  — 向量化加权求和
            # alpha_k_arr[:, None, None] broadcast 到 (C, H, W)
            weights_sum = np.sum(alpha_k_arr[:, None, None] * activations, axis=0)  # (H, W)

            heatmap = np.maximum(0.0, weights_sum)   # ReLU
            heatmap = self._normalize_heatmap(heatmap)

            return heatmap

        finally:
            self._remove_hooks()

    def _compute_weights(self, gradients: np.ndarray) -> Dict[int, float]:
        """
        计算神经元重要性权重α

        α_k^c = GlobalAveragePool(∂Y^c/∂A^k) = mean(gradients[k])

        Args:
            gradients: 梯度张量 (C,) 或 (C, H, W)

        Returns:
            每个通道的权重字典
        """
        weights = {}

        for k in range(gradients.shape[0]):
            weights[k] = float(np.mean(gradients[k]))
        return weights

    def _normalize_heatmap(self, heatmap: np.ndarray) -> np.ndarray:
        """
        归一化热力图到[0,1]

        Args:
            heatmap: 原始热力图

        Returns:
            归一化后的热力图
        """
        heatmap = heatmap.astype(np.float32)

        min_val = np.min(heatmap)
        max_val = np.max(heatmap)

        if max_val > 1e-8:
            # 只需最大值非零即可归一化；min 在 ReLU 之后必然 >= 0
            heatmap = heatmap / max_val
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap

    def generate_batch(
        self,
        input_tensors: List[Tensor],
        class_indices: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        批量生成Grad-CAM热力图

        Args:
            input_tensors: 输入张量列表
            class_indices: 目标类别索引列表，如果为None则使用预测类别

        Returns:
            热力图列表
        """
        heatmaps = []

        for i, input_tensor in enumerate(input_tensors):
            class_idx = None
            if class_indices is not None:
                class_idx = class_indices[i]

            heatmap = self.generate(input_tensor, class_idx)
            heatmaps.append(heatmap)

        return heatmaps

    def __call__(
        self,
        input_tensor: Tensor,
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        调用generate方法的便捷方式

        Args:
            input_tensor: 输入张量
            class_idx: 目标类别索引

        Returns:
            热力图
        """
        return self.generate(input_tensor, class_idx)


def get_all_conv_layers(model: Module) -> List[tuple]:
    """
    获取模型中所有卷积层

    Args:
        model: 神经网络模型

    Returns:
        卷积层列表 [(name, layer), ...]
    """
    conv_layers = []
    visited = set()

    def search_modules(module, prefix='', depth=0):
        if depth > 20 or id(module) in visited:
            return

        visited.add(id(module))

        for name, child in module.__dict__.items():
            if name.startswith('_'):
                continue

            if isinstance(child, Layer):
                module_name = type(child).__name__
                if 'conv' in module_name.lower() or 'Conv' in module_name:
                    full_name = f"{prefix}.{name}" if prefix else name
                    conv_layers.append((full_name, child))
            elif hasattr(child, '__dict__') and not name.startswith('_'):
                new_prefix = f"{prefix}.{name}" if prefix else name
                search_modules(child, new_prefix, depth + 1)

    search_modules(model)
    return conv_layers


def suggest_target_layer(model: Module) -> Optional[Layer]:
    """
    自动建议最佳的Grad-CAM目标层

    Args:
        model: 神经网络模型

    Returns:
        建议的目标层，如果找不到则返回None
    """
    conv_layers = get_all_conv_layers(model)

    if not conv_layers:
        return None

    return conv_layers[-1][1]


def create_gradcam(
    model: Module,
    target_layer: Optional[Layer] = None
) -> GradCAM:
    """
    创建GradCAM实例的便捷工厂函数

    Args:
        model: 神经网络模型
        target_layer: 目标层，如果为None则自动选择

    Returns:
        GradCAM实例
    """
    if target_layer is None:
        target_layer = suggest_target_layer(model)

    if target_layer is None:
        raise ValueError(
            "无法自动找到合适的卷积层。"
            "请手动指定 target_layer 参数。"
        )

    return GradCAM(model, target_layer)