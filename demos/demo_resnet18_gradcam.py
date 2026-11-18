#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ResNet18 + Grad-CAM 可视化演示
基于 DonkeyCar 自动驾驶数据集，展示不同转向类别和不同深度的热力图
"""
import sys, os
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.ndimage import zoom
import cv2

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients

# ============================================================
# 加载模型
# ============================================================
def load_model():
    # 尝试加载预训练权重
    model_path = project_root / 'tests' / 'test_donkey' / 'results' / 'model_200.json'
    sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))
    from model import ResNet18AutoDrive
    model = ResNet18AutoDrive()
    if model_path.exists():
        serializer = Serializer()
        serializer.load(model, str(model_path))
        print(f'[GradCAM] Loaded model from {model_path}')
    else:
        print('[GradCAM] No pretrained model found, using random weights')
    return model


def get_target_layer(model):
    """返回 layer4 最后一个残差块的 conv2 作为目标层"""
    return model.layer4.layers[1].conv2


def get_layer_by_depth(model):
    """返回不同深度的卷积层"""
    return {
        'stem (conv1)': model.conv1,
        'layer1.block2.conv2': model.layer1.layers[1].conv2,
        'layer2.block2.conv2': model.layer2.layers[1].conv2,
        'layer3.block2.conv2': model.layer3.layers[1].conv2,
        'layer4.block2.conv2': model.layer4.layers[1].conv2,
    }


def classify_angle(angle):
    if angle < -0.15: return '大幅度左转'
    elif angle < -0.05: return '小幅左转'
    elif angle <= 0.05: return '直行'
    elif angle <= 0.15: return '小幅右转'
    else: return '大幅度右转'


def generate_heatmap(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图计算"""
    fh, fd = capture_features(target_layer)
    registry = HookRegistry()
    registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is not None and 'BatchNorm' not in type(grad_layer).__name__:
        pass
    else:
        grad_layer = target_layer

    gh, gd = capture_gradients(grad_layer)
    output_scalar = output[0, 0]
    output_scalar.backward()

    try:
        activations = fd['output'][0]
        gradients = gd['grad_output'][0]
        if gradients.ndim == 3:
            alpha = gradients.mean(axis=(1, 2))
        elif gradients.ndim == 2:
            alpha = gradients.mean(axis=1)
        else:
            alpha = gradients
        cam = np.sum(alpha[:, None, None] * activations, axis=0)
        cam = np.maximum(0.0, cam)
        mx = np.max(cam)
        if mx > 1e-8:
            cam = cam / mx
        return cam
    finally:
        fh.remove()
        gh.remove()


def plot_depth_comparison(model, val_dataset, out_dir):
    """不同深度层的 Grad-CAM 对比"""
    print('[GradCAM] Generating depth comparison...')
    layers = get_layer_by_depth(model)
    # 选一个样本
    img, label = val_dataset[20]
    true_angle = label[0]
    cat = classify_angle(true_angle)
    input_tensor = Tensor(img[np.newaxis, ...])
    output = model(input_tensor)
    pred_angle = float(output.data[0, 0])

    orig = img.transpose(1, 2, 0)
    orig = np.clip(orig, 0, 1)

    names = list(layers.keys())
    n = len(names)
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 8.5))
    fig.suptitle(f'ResNet18 不同深度层 Grad-CAM 对比\n类别: {cat}  |  真实角度: {true_angle:.4f}  |  预测角度: {pred_angle:.4f}',
                 fontsize=13, fontweight='bold')

    for j, name in enumerate(names):
        hm = generate_heatmap(model, layers[name], input_tensor)
        if hm is not None:
            ax0 = axes[0, j] if n > 1 else axes[0]
            im = ax0.imshow(hm, cmap='jet', interpolation='bilinear', vmin=0, vmax=1)
            ax0.set_title(f'{name}\n({hm.shape[0]}x{hm.shape[1]})', fontsize=9)
            ax0.axis('off')
            plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

            # Overlay
            zf = (orig.shape[0] / hm.shape[0], orig.shape[1] / hm.shape[1])
            hm_rs = zoom(hm, zf, order=1)
            hm_color = np.array(plt.cm.jet(hm_rs))[:, :, :3]
            overlay = 0.5 * orig + 0.5 * hm_color
            ax1 = axes[1, j] if n > 1 else axes[1]
            ax1.imshow(np.clip(overlay, 0, 1))
            ax1.set_title('Overlay', fontsize=9)
            ax1.axis('off')

    plt.tight_layout()
    path = out_dir / 'resnet18_gradcam_depth.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def plot_category_comparison(model, val_dataset, out_dir):
    """不同转向类别的 Grad-CAM 对比"""
    print('[GradCAM] Generating category comparison...')
    target = get_target_layer(model)

    # 每种类别选一个样本
    categories = {'大幅度左转': [], '小幅左转': [], '直行': [], '小幅右转': [], '大幅度右转': []}
    for i in range(len(val_dataset)):
        if hasattr(val_dataset, 'file_list'):
            angle = val_dataset.file_list[i][1]
        else:
            angle = val_dataset[i][1][0]
        cat = classify_angle(float(angle))
        if len(categories[cat]) < 1:
            categories[cat].append(i)
    samples = [(idx, cat) for cat, idxs in categories.items() for idx in idxs]

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4.2 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (idx, cat) in enumerate(samples):
        img, label = val_dataset[idx]
        true_angle = label[0]
        inp = Tensor(img[np.newaxis, ...])
        out = model(inp)
        pred = float(out.data[0, 0])
        hm = generate_heatmap(model, target, inp)

        orig = np.clip(img.transpose(1, 2, 0), 0, 1)
        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f'{cat}\nTrue: {true_angle:.4f} / Pred: {pred:.4f}', fontsize=10)
        axes[i, 0].axis('off')

        if hm is not None:
            axes[i, 1].imshow(hm, cmap='jet', interpolation='bilinear', vmin=0, vmax=1)
            axes[i, 1].set_title(f'Grad-CAM (l4b.conv2)', fontsize=10)
            axes[i, 1].axis('off')

            zf = (orig.shape[0] / hm.shape[0], orig.shape[1] / hm.shape[1])
            hm_rs = zoom(hm, zf, order=1)
            hm_c = np.array(plt.cm.jet(hm_rs))[:, :, :3]
            ov = 0.5 * orig + 0.5 * hm_c
            axes[i, 2].imshow(np.clip(ov, 0, 1))
            axes[i, 2].set_title('Overlay', fontsize=10)
            axes[i, 2].axis('off')

    fig.suptitle('ResNet18 Grad-CAM: 不同转向类别对比 (layer4.block2.conv2)', fontsize=14, fontweight='bold', y=1.005)
    plt.tight_layout()
    path = out_dir / 'resnet18_gradcam_categories.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


def main():
    out_dir = script_dir / 'output'
    out_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('ResNet18 Grad-CAM 可视化演示')
    print('=' * 60)

    model = load_model()

    # 加载验证集
    from dataset import AutoDriveDataset, preprocess_image
    val_dataset = AutoDriveDataset(mode='val', transform=preprocess_image)
    print(f'Validation samples: {len(val_dataset)}')

    # 查看模型结构
    all_convs = get_all_conv_layers(model)
    print(f'Total conv layers: {len(all_convs)}')
    target = get_target_layer(model)
    print(f'Target layer: {type(target).__name__}, out_channels={target.out_channels}')

    plot_depth_comparison(model, val_dataset, out_dir)
    plot_category_comparison(model, val_dataset, out_dir)

    print('\n=== Grad-CAM demo complete! ===')


if __name__ == '__main__':
    main()
