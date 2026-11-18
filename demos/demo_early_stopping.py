#!/usr#!/usr/bin/env python
# -*- coding: utf-8#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 Donkey#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root =#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / '#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100,#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train,#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停的#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停的模型（模拟多个 epoch 收集数据）
    model = ResNet18AutoDrive()
    loss#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停的模型（模拟多个 epoch 收集数据）
    model = ResNet18AutoDrive()
    loss_fn = MSELoss()
    optimizer = Adam(model.get_params_list(only_trainable=True),#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停的模型（模拟多个 epoch 收集数据）
    model = ResNet18AutoDrive()
    loss_fn = MSELoss()
    optimizer = Adam(model.get_params_list(only_trainable=True), lr=0.001)

    # 早停参数：patience=5, mode#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停的模型（模拟多个 epoch 收集数据）
    model = ResNet18AutoDrive()
    loss_fn = MSELoss()
    optimizer = Adam(model.get_params_list(only_trainable=True), lr=0.001)

    # 早停参数：patience=5, mode='loss'
    patience = 5
    best_loss = float('inf')
    epochs_no#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停的模型（模拟多个 epoch 收集数据）
    model = ResNet18AutoDrive()
    loss_fn = MSELoss()
    optimizer = Adam(model.get_params_list(only_trainable=True), lr=0.001)

    # 早停参数：patience=5, mode='loss'
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0
    best_#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停的模型（模拟多个 epoch 收集数据）
    model = ResNet18AutoDrive()
    loss_fn = MSELoss()
    optimizer = Adam(model.get_params_list(only_trainable=True), lr=0.001)

    # 早停参数：patience=5, mode='loss'
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None

    # 存储每个 epoch#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停的模型（模拟多个 epoch 收集数据）
    model = ResNet18AutoDrive()
    loss_fn = MSELoss()
    optimizer = Adam(model.get_params_list(only_trainable=True), lr=0.001)

    # 早停参数：patience=5, mode='loss'
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None

    # 存储每个 epoch 的指标
    history = {'train_loss': [], 'val_loss': [], 'stopped#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停的模型（模拟多个 epoch 收集数据）
    model = ResNet18AutoDrive()
    loss_fn = MSELoss()
    optimizer = Adam(model.get_params_list(only_trainable=True), lr=0.001)

    # 早停参数：patience=5, mode='loss'
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None

    # 存储每个 epoch 的指标
    history = {'train_loss': [], 'val_loss': [], 'stopped_epoch': None}

    print(f'\n#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停的模型（模拟多个 epoch 收集数据）
    model = ResNet18AutoDrive()
    loss_fn = MSELoss()
    optimizer = Adam(model.get_params_list(only_trainable=True), lr=0.001)

    # 早停参数：patience=5, mode='loss'
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None

    # 存储每个 epoch 的指标
    history = {'train_loss': [], 'val_loss': [], 'stopped_epoch': None}

    print(f'\n早停参数: patience={patience}, mode=loss (越小越好)')
    print(f#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停的模型（模拟多个 epoch 收集数据）
    model = ResNet18AutoDrive()
    loss_fn = MSELoss()
    optimizer = Adam(model.get_params_list(only_trainable=True), lr=0.001)

    # 早停参数：patience=5, mode='loss'
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None

    # 存储每个 epoch 的指标
    history = {'train_loss': [], 'val_loss': [], 'stopped_epoch': None}

    print(f'\n早停参数: patience={patience}, mode=loss (越小越好)')
    print(f'模拟训练过程 (实际每个 epoch 仅#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停的模型（模拟多个 epoch 收集数据）
    model = ResNet18AutoDrive()
    loss_fn = MSELoss()
    optimizer = Adam(model.get_params_list(only_trainable=True), lr=0.001)

    # 早停参数：patience=5, mode='loss'
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None

    # 存储每个 epoch 的指标
    history = {'train_loss': [], 'val_loss': [], 'stopped_epoch': None}

    print(f'\n早停参数: patience={patience}, mode=loss (越小越好)')
    print(f'模拟训练过程 (实际每个 epoch 仅 200 个样本做快速演示):\n')

    # 模拟约 20 个 epoch 的训练过程
    MAX_#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停的模型（模拟多个 epoch 收集数据）
    model = ResNet18AutoDrive()
    loss_fn = MSELoss()
    optimizer = Adam(model.get_params_list(only_trainable=True), lr=0.001)

    # 早停参数：patience=5, mode='loss'
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None

    # 存储每个 epoch 的指标
    history = {'train_loss': [], 'val_loss': [], 'stopped_epoch': None}

    print(f'\n早停参数: patience={patience}, mode=loss (越小越好)')
    print(f'模拟训练过程 (实际每个 epoch 仅 200 个样本做快速演示):\n')

    # 模拟约 20 个 epoch 的训练过程
    MAX_EPOCHS = 20
    batch_size#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
早停机制演示
使用 DonkeyCar + ResNet18，展示 EarlyStopping 的训练曲线和触发过程
训练 1 Epoch（速度优先），通过模拟多个 epoch 展示早停效果
"""
import sys, os, time
import numpy as np
from pathlib import Path

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'code'))
sys.path.insert(0, str(project_root / 'tests' / 'test_donkey'))

from eneuro.base import Tensor, Config
from eneuro.nn.optim import Adam
from eneuro.nn.loss import MSELoss
from dataset import AutoDriveDataset, preprocess_image
from model import ResNet18AutoDrive


def run_with_early_stop(out_dir):
    """运行早停演示"""
    print('=' * 70)
    print(' Early Stopping 早停机制演示')
    print('=' * 70)

    # 准备数据
    train_ds = AutoDriveDataset(mode='train', transform=preprocess_image)
    val_ds = AutoDriveDataset(mode='val', transform=preprocess_image)
    n = min(300, len(train_ds))
    vn = min(100, len(val_ds))

    print(f'\n训练集: {n} samples | 验证集: {vn} samples')

    # 准备 batch 数据
    x_train, y_train = [], []
    for i in range(n):
        img, label = train_ds[i]
        x_train.append(img)
        y_train.append(label)
    x_train = np.stack(x_train)
    y_train = np.stack(y_train).reshape(-1, 1)

    x_val, y_val = [], []
    for i in range(vn):
        img, label = val_ds[i]
        x_val.append(img)
        y_val.append(label)
    x_val = np.stack(x_val)
    y_val = np.stack(y_val).reshape(-1, 1)

    # 构建带早停的模型（模拟多个 epoch 收集数据）
    model = ResNet18AutoDrive()
    loss_fn = MSELoss()
    optimizer = Adam(model.get_params_list(only_trainable=True), lr=0.001)

    # 早停参数：patience=5, mode='loss'
    patience = 5
    best_loss = float('inf')
    epochs_no_improve = 0
    best_weights = None

    # 存储每个 epoch 的指标
    history = {'train_loss': [], 'val_loss': [], 'stopped_epoch': None}

    print(f'\n早停参数: patience={patience}, mode=loss (越小越好)')
    print(f'模拟训练过程 (实际每个 epoch 仅 200 个样本做快速演示):\n')

    # 模拟约 20 个 epoch 的训练过程
    MAX_EPOCHS = 20
    batch_size = 32
    actual_epochs = 0

    for epoch in range(MAX