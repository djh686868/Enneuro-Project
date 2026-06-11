import numpy as np
import cv2
import random
from ..base import Tensor, as_Tensor, as_array

def normalize(tensor: Tensor, mean, std) -> Tensor:
    #标准化: (x - mean) / std
    data = tensor.data.copy()
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    if data.ndim == 3 and data.shape[-1] == len(mean):
        mean = mean.reshape(1, 1, -1)
        std = std.reshape(1, 1, -1)
    normalized = (data - mean) / (std + 1e-8)
    return Tensor(normalized, requires_grad=tensor.requires_grad)


def to_range(tensor: Tensor, min_out=0, max_out=1) -> Tensor:
    #将数据缩放到 [min_out, max_out] 区间
    data = tensor.data
    min_val, max_val = data.min(), data.max()
    if max_val - min_val < 1e-8:
        scaled = np.full_like(data, min_out)
    else:
        scaled = (data - min_val) / (max_val - min_val) * (max_out - min_out) + min_out
    return Tensor(scaled, requires_grad=tensor.requires_grad)


#几何变换
def hflip(tensor: Tensor, p=0.5) -> Tensor:
    #随机水平翻转
    if random.random() < p:
        flipped = np.fliplr(tensor.data).copy()
        return Tensor(flipped, requires_grad=tensor.requires_grad)
    return tensor


def vflip(tensor: Tensor, p=0.5) -> Tensor:
    #随机垂直翻转
    if random.random() < p:
        flipped = np.flipud(tensor.data).copy()
        return Tensor(flipped, requires_grad=tensor.requires_grad)
    return tensor


def rotate(tensor: Tensor, angle=None, p=0.5) -> Tensor:
    #随机旋转（角度范围默认 -15~15 度）
    if random.random() < p:
        img = tensor.data
        h, w = img.shape[:2]
        if angle is None:
            angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h))
        return Tensor(rotated, requires_grad=tensor.requires_grad)
    return tensor


def random_crop(tensor: Tensor, size, p=0.5) -> Tensor:
    #随机裁剪到指定大小 (h, w)
    if random.random() < p:
        img = tensor.data
        h, w = img.shape[:2]
        crop_h, crop_w = size
        if crop_h > h or crop_w > w:
            raise ValueError(f"Crop size {size} > image size {(h,w)}")
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        cropped = img[top:top+crop_h, left:left+crop_w].copy()
        return Tensor(cropped, requires_grad=tensor.requires_grad)
    return tensor


#色彩增强
def adjust_hsv(tensor: Tensor, h_gain=0.5, s_gain=0.5, v_gain=0.5, p=0.5) -> Tensor:
    #随机调整 HSV（要求输入为 RGB 图像，uint8）
    if random.random() < p:
        img = tensor.data.astype(np.float32)
        r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:,:,0] = (hsv[:,:,0] * r[0]) % 180
        hsv[:,:,1] = np.clip(hsv[:,:,1] * r[1], 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * r[2], 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return Tensor(img, requires_grad=tensor.requires_grad)
    return tensor


def gaussian_blur(tensor: Tensor, kernel_size=(5,5), sigma=0, p=0.5) -> Tensor:
    #高斯模糊
    if random.random() < p:
        blurred = cv2.GaussianBlur(tensor.data, kernel_size, sigma)
        return Tensor(blurred, requires_grad=tensor.requires_grad)
    return tensor


def random_noise(tensor: Tensor, mean=0, std=25, p=0.5) -> Tensor:
    #添加高斯噪声
    if random.random() < p:
        noise = np.random.normal(mean, std, tensor.data.shape)
        noisy = tensor.data + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return Tensor(noisy, requires_grad=tensor.requires_grad)
    return tensor


#组合函数
def compose(*funcs):
    #组合多个变换函数，依次应用
    def applied(tensor):
        for f in funcs:
            tensor = f(tensor)
        return tensor
    return applied
