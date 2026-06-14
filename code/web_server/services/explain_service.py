"""GradCAM 和全通道特征图服务。"""
import base64
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

import cv2
from eneuro.explainability.gradcam import GradCAM
from eneuro.utils.hooks import capture_features
from web_server.services.model_registry import get_model


def _b64_png(arr_hw: np.ndarray) -> str:
    img_u8 = (arr_hw * 255).clip(0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", img_u8)
    return base64.b64encode(buf).decode()


def _get_model_in_channels(model) -> int:
    """从模型第一个卷积层读取 in_channels，默认返回 1。"""
    for attr in ('stem_conv', 'conv1'):
        layer = getattr(model, attr, None)
        if layer is not None and hasattr(layer, 'in_channels'):
            return int(layer.in_channels)
    for v in vars(model).values():
        if hasattr(v, 'in_channels'):
            return int(v.in_channels)
    return 1


def _parse_image(b64_str: str, in_channels: int = 1) -> np.ndarray:
    data = base64.b64decode(b64_str)
    arr = np.frombuffer(data, dtype=np.uint8)
    if in_channels == 1:
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Failed to decode image")
        img = cv2.resize(img, (32, 32))
        x = img.astype(np.float32) / 255.0
        return x[np.newaxis, np.newaxis, :, :]    # (1, 1, H, W)
    else:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 120))          # 保持 DonkeyCar 原始宽高比
        x = img.astype(np.float32) / 255.0
        return x.transpose(2, 0, 1)[np.newaxis]   # (1, 3, H, W)


def _find_layer(model, name: str):
    if hasattr(model, name):
        return getattr(model, name)
    if name.startswith("layer_") and hasattr(model, "layers"):
        idx = int(name.split("_")[1])
        return model.layers[idx]
    # 尝试数字索引
    try:
        idx = int(name)
        if hasattr(model, "layers"):
            return model.layers[idx]
    except ValueError:
        pass
    raise ValueError(f"Layer '{name}' not found in model. "
                     f"Try 'layer_0', 'layer_1', or attribute name.")


def run_gradcam(model_id: str, image_b64: str, layer_name: str, class_idx=None):
    from eneuro.base.core import Tensor
    model = get_model(model_id)
    in_ch = _get_model_in_channels(model)
    img_np = _parse_image(image_b64, in_channels=in_ch)
    x = Tensor(img_np, requires_grad=False)

    target_layer = _find_layer(model, layer_name)
    cam = GradCAM(model, target_layer)
    heatmap = cam.generate(x, class_idx=class_idx)   # (H, W) float[0,1]

    DISPLAY = 224
    heatmap_big = cv2.resize(heatmap, (DISPLAY, DISPLAY), interpolation=cv2.INTER_LINEAR)
    h_color = cv2.applyColorMap((heatmap_big * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 原始图像叠加：灰度/彩色均处理
    if in_ch == 1:
        orig_u8 = (img_np[0, 0] * 255).astype(np.uint8)
        orig_bgr = cv2.cvtColor(cv2.resize(orig_u8, (DISPLAY, DISPLAY)), cv2.COLOR_GRAY2BGR)
    else:
        orig_rgb = (img_np[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        orig_bgr = cv2.cvtColor(cv2.resize(orig_rgb, (DISPLAY, DISPLAY)), cv2.COLOR_RGB2BGR)

    overlay = cv2.addWeighted(orig_bgr, 0.5, h_color, 0.5, 0)

    _, buf_h = cv2.imencode(".png", h_color)
    _, buf_o = cv2.imencode(".png", overlay)
    return {
        "heatmap": base64.b64encode(buf_h).decode(),
        "overlay": base64.b64encode(buf_o).decode(),
    }


def run_feature_maps(model_id: str, image_b64: str, layer_name: str):
    from eneuro.base.core import Tensor
    model = get_model(model_id)
    in_ch = _get_model_in_channels(model)
    img_np = _parse_image(image_b64, in_channels=in_ch)
    x = Tensor(img_np, requires_grad=False)

    target_layer = _find_layer(model, layer_name)
    hook = capture_features(target_layer)

    model(x)

    acts = getattr(target_layer, "_captured_features", None)
    if acts is None:
        raise RuntimeError("No features captured — check layer name")

    if hasattr(acts, "data"):
        acts = acts.data
    try:
        acts = acts.get()   # cupy → numpy
    except AttributeError:
        pass

    acts = acts[0]  # (C, H, W)
    maps = []
    for c in range(acts.shape[0]):
        ch = acts[c]
        ch_norm = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
        maps.append({"channel": c, "image": _b64_png(ch_norm)})

    return {"maps": maps, "num_channels": len(maps)}
