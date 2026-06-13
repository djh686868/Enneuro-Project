# -*- coding: utf-8 -*-
import sys
from pathlib import Path

import numpy as np

# 确保能导入 code 下的包
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / 'code'))

from eneuro.base import as_Tensor, Config  # noqa: E402
from eneuro.base import functions as F      # noqa: E402
from eneuro.nn.module import (
    Module,
    Conv2d,
    BatchNorm,
    Linear,
    ResidualBlock,
    Sequential,
)  # noqa: E402
from eneuro.ao import GraphOptimizer, model_to_graph, graph_to_executor, graph_apply_fuse, graph_apply_cast, GradScaler
from eneuro.nn.optim import SGD, Adam
from eneuro.nn.loss import crossEntropyError

def progress_bar(current, total, epoch, loss, acc, width=30): # 进度条
    """current: 当前已处理样本数；total: 总样本数"""
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    sys.stdout.write(
        f'\rEpoch {epoch+1:3d} |{bar}| {percent*100:5.1f}% ({current:5}/{total:5}) '
        f' | loss={loss:.4f} | acc={acc:.3f}'
    )
    sys.stdout.flush()

class ResNet18(Module):
    """基于项目中 `ResidualBlock` 的 ResNet-18 风格微型实现（用于测试）。"""

    def __init__(self, num_classes=1000):
        super().__init__()
        # 初始通道数基准
        self.in_channels = 64

        # stem: 7x7 conv, stride=2, pad=3
        self.conv1 = Conv2d(64, kernel_size=7, stride=2, pad=3, in_channels=3)
        self.bn1 = BatchNorm(64)

        # 4 个 stage，每个 stage 包含若干个 ResidualBlock
        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

        # 分类头
        self.fc = Linear(num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        # 首个 block 可能需要下采样/升维
        downsample = (stride != 1) or (self.in_channels != out_channels)
        layers.append(ResidualBlock(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels

        for _ in range(blocks - 1):
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=1, downsample=False))

        return Sequential(*layers)

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.pooling(x, kernel_size=3, stride=2, pad=1)

        # stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # global pool -> fc
        x = F.global_average_pooling(x)
        x = F.flatten(x)
        x = self.fc(x)
        return x


def test_resnet18_forward_backward():
    print_gpu_mem()
    
    from eneuro.data import DataLoader
    from eneuro.base.functions import has_cupy
    import time
    #'''
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir / 'data'
    dataset = SteeringDataset(
        root_dir=str(data_path),
        transform=lambda x: x.transpose(2, 0, 1)   # 可选：将 (H,W,C) 转为 (C,H,W)
    )
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #'''
    
    model = ResNet18(num_classes=20)
    model.to('cuda' if has_cupy else 'cpu')

    # optimizer = SGD(model.params())
    optimizer = Adam(model.params(), lr=1e-4)
    
    loss_fn = crossEntropyError

    tic = time.time()
    for batch_idx, (images, labels) in enumerate(dataloader):
    #for i in range(len(dataset)):
        '''
        sample_image, sample_label = dataset[0]
        sample_image = np.ones_like(sample_image)
        sample_label = np.ones_like(sample_label)

        batch_data = [sample_image for i in range(batch_size)]
        batch_target = [sample_label for i in range(batch_size)]

        sample_image = np.stack(batch_data)
        sample_label = np.stack(batch_target)

        batch_idx, (images, labels) = (i,(sample_image, sample_label))
        images = as_Tensor(images)
        #'''

        y_pre = model(images)
        
        loss = y_pre * y_pre
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        break
    toc = time.time()
    duration = toc - tic
    print(f"[PASS] test_resnet18_forward_backward {duration:.4f}")
    #import gc
    #gc.collect()
    print_gpu_mem()

# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

from eneuro.base import as_Tensor
from eneuro.data import Dataset


class SteeringDataset(Dataset):
    """
    从目录读取转向角图片的数据集。
    文件名格式：`id_steeringAngle.jpg`，例如 0_0.0667.jpg、1_-0.0222.jpg。
    图片尺寸：160x120（不强制检查，但假设所有图片均为此尺寸）。
    标签映射：将 steeringAngle (-1.0 到 1.0) 划分为 20 个连续区间，
              每个区间宽度 0.1，返回对应的类别索引 (0~19)。
    """

    def __init__(self, root_dir, transform=None, target_transform=None, train=True):
        """
        参数:
            root_dir: str 或 Path，包含图片的根目录。
            transform: 可调用对象，接收已加载的图片数组 (H,W,C, np.float32, 范围[0,1])，
                       返回处理后的数据。默认 None 表示恒等变换。
            target_transform: 可调用对象，接收原始转向角 (float)，
                             返回任意目标值。若为 None，则自动使用默认的类别映射。
            train: bool，兼容基类，本实现中未使用。
        """
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.images = []      # 存储图片完整路径
        self.angles = []      # 存储原始转向角 (float)
        self.prepare()

        # 若用户未提供 target_transform，则使用默认的类别映射
        if target_transform is None:
            target_transform = self._default_target_transform
        super().__init__(train, transform, target_transform)

    def prepare(self):
        """扫描目录，解析所有 .jpg 文件，收集有效路径与转向角。"""
        if not self.root_dir.is_dir():
            raise NotADirectoryError(f"目录不存在: {self.root_dir}")

        pattern = re.compile(r'^(\d+)_([-+]?\d*\.?\d+)\.jpg$')
        for fname in os.listdir(self.root_dir):
            #print(fname)
            if not fname.lower().endswith('.jpg'):
                continue
            match = pattern.match(fname)
            if not match:
                continue    # 跳过不符合命名规范的文件
            angle_str = match.group(2)
            try:
                angle = float(angle_str)
            except ValueError:
                continue    # 转换失败则跳过
            #print(f"{match.group(1)} {angle}")

            # 可选：只接受 -1.0 <= angle <= 1.0 范围内的样本
            if angle < -1.0 or angle > 1.0:
                continue

            img_path = self.root_dir / fname
            self.images.append(str(img_path))
            self.angles.append(angle)

        if not self.images:
            raise RuntimeError(f"在 {self.root_dir} 中未找到符合命名规范且角度在 [-1,1] 内的图片。")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        返回:
            img: eneuro.base.Tensor，形状 (C, H, W) 或 (H, W, C) 取决于 transform，
                 默认是 (H, W, C) 的 float32 数组（范围 [0,1]）转换为 Tensor。
            label: eneuro.base.Tensor，标量整数，类别索引 0~19。
        """
        img_path = self.images[index]
        angle = self.angles[index]

        # 1. 读取图片并转为 RGB，归一化到 [0,1]
        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert('RGB')
            img = np.array(pil_img, dtype=np.float32) / 255.0   # (H, W, C)

        # 2. 应用用户提供的 transform（例如标准化、维度重排等）
        if self.transform is not None:
            img = self.transform(img)

        # 3. 计算标签（通过 target_transform）
        if self.target_transform is not None:
            label = self.target_transform(angle)
        else:
            # 实际上 __init__ 已保证 target_transform 不为 None，此分支仅防御
            label = self._default_target_transform(angle)

        #'''
        # 4. 转换为框架要求的 Tensor 类型
        img_tensor = as_Tensor(img)
        # 确保 label 是标量 Tensor
        if not isinstance(label, np.generic) and not isinstance(label, (int, np.integer)):
            # 若用户自定义 target_transform 返回了 numpy/Tensor 等，直接转换
            label_tensor = as_Tensor(label)
        else:
            label_tensor = as_Tensor(np.array(label, dtype=np.int64))
        # 若 label 是多维，挤压为标量
        if label_tensor.ndim == 0:
            pass
        elif label_tensor.size == 1:
            label_tensor = label_tensor.reshape(())
        else:
            raise ValueError(f"标签应为标量，但得到形状 {label_tensor.shape}")
        #'''
        #img_tensor = img
        #label_tensor = label

        return img_tensor, label_tensor

    @staticmethod
    def _default_target_transform(angle):
        """
        将角度 (-1.0, 1.0] 映射到类别索引 0~19。
        区间划分：[-1.0, -0.9), [-0.9, -0.8), ..., [0.9, 1.0]。
        其中 -1.0 归为 0，1.0 归为 19。
        """
        num_classes = 20
        low, high = -1.0, 1.0
        bin_width = (high - low) / num_classes   # = 0.1
        idx = int((angle - low) // bin_width)
        if idx == num_classes:   # 处理 angle == 1.0 的边界情况
            idx = num_classes - 1
        # 钳位，防止因浮点误差导致越界
        idx = max(0, min(idx, num_classes - 1))
        return idx

def print_gpu_mem():
    from eneuro.base.functions import has_cupy
    if has_cupy:
        import cupy as cp
        free, total = cp.cuda.Device().mem_info
        used = total - free
        print(f"GPU memory: used={used/1024**2:.1f}MB, free={free/1024**2:.1f}MB, total={total/1024**2:.1f}MB")

def train(num_epoch=10, option='normal', autocast=False):
    print_gpu_mem()

    batch_size = 16

    from eneuro.base.functions import has_cupy
    from eneuro.data import DataLoader
    import time
    # 实例化数据集
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir / 'data'
    dataset = SteeringDataset(
        root_dir=str(data_path),
        transform=lambda x: x.transpose(2, 0, 1)   # 可选：将 (H,W,C) 转为 (C,H,W)
    )

    # 配合 DataLoader 使用
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ResNet18(num_classes=20)
    model.to('cuda' if has_cupy else 'cpu')
    # optimizer = SGD(model.params())
    optimizer = Adam(model.params(), lr=1e-4)
    
    loss_fn = crossEntropyError
    if autocast:
        scaler = GradScaler()

    if option in ['graph', 'optim_graph']:
        for batch_idx, (images, labels) in enumerate(dataloader):
            sample_input = images
            if option == 'graph':
                graph = model_to_graph(model,sample_input=sample_input)
            elif option == 'optim_graph':
                graph = graph_apply_fuse(model_to_graph(model,sample_input=sample_input))

            if autocast:
                graph = graph_apply_cast(graph)
                #graph.visualize()
            
            executor = graph_to_executor(graph)
            break
    elif autocast:
        print("autocast should be used with graph!")
        return 1e-5

    for epoch in range(num_epoch):
        tic = time.time()
        loss_sum, acc_sum, sample_num = 0., 0, 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            # images: Tensor (B, C, H, W) 或 (B, H, W, C)
            # labels: Tensor (B,)

            if images.shape[0] != batch_size:
                break

            if autocast:
                y_pre = executor.forward(images)
                loss = loss_fn(y_pre, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)

            else:
                if option == 'normal':
                    y_pre = model(images) # (B, num_classes)
                elif option in ['graph', 'optim_graph']:
                    y_pre = executor.forward(images)
                loss = loss_fn(y_pre, labels)
            
                loss.backward()
                optimizer.step()

            optimizer.zero_grad()
            
            #assert isinstance(y_pre, Tensor)
            pred_classes = np.argmax(y_pre.to('cpu').data, axis=1)
            batch_acc = np.mean(pred_classes == labels.to('cpu').data)

            loss_sum += loss.to('cpu').data * len(images)
            if not np.isnan(batch_acc):
                acc_sum += batch_acc * len(images)
            sample_num += len(images)

            display_acc = (acc_sum / sample_num) if sample_num > 0 and not np.isnan(acc_sum) else np.nan
            progress_bar(batch_idx * batch_size + len(images), len(dataloader.dataset), epoch, loss.to('cpu').data, display_acc)

            #break
        
        toc = time.time()
        duration = toc - tic
        print(f"\nEpoch completed in {duration:.4f}s\n")

    #del model
    #del optimizer
    #import gc
    #gc.collect()
    print_gpu_mem()
    return duration

def test1():
    normal_t = train(num_epoch=1, option='normal', autocast=False)
    print(f"normal training complete in {normal_t:.4f}s")

def test2():
    normal_t = train(num_epoch=1, option='graph', autocast=False)
    print(f"graph training complete in {normal_t:.4f}s")

def test3():
    cast_t = train(num_epoch=1, option='graph', autocast=True)
    print(f"autocast graph training complete in {cast_t:.4f}s")

def test4():
    normal_t = train(num_epoch=1, option='optim_graph', autocast=False)
    print(f"optim_graph training complete in {normal_t:.4f}s")

def test5():
    cast_t = train(num_epoch=1, option='optim_graph', autocast=True)
    print(f"autocast optim_graph training complete in {cast_t:.4f}s")

if __name__ == '__main__':
    
    '''
    for i in range(1000):
        test_resnet18_forward_backward()
    #'''

    #'''
    for i in range(1):
        test1()
        # test2()
        # test3()
        # test4()
        # test5()
    #'''

    '''
    for i in range(10):
        test2()
    #'''
