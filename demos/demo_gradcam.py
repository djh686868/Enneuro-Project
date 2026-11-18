# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/g# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  dem# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' /# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from en# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_don# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, gra# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.maked# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3]# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.con# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.l# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0,# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1))# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[1,j].axis('off')

    axes[0,0].set_ylabel('Heatmap# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[1,j].axis('off')

    axes[0,0].set_ylabel('Heatmap', fontsize=13, fontweight='bold')
    axes[1,0].set_ylabel('Overlay',  fontsize=13# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[1,j].axis('off')

    axes[0,0].set_ylabel('Heatmap', fontsize=13, fontweight='bold')
    axes[1,0].set_ylabel('Overlay',  fontsize=13, fontweight='bold')
    fig.suptitle('ResNet18 Grad-CAM:# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[1,j].axis('off')

    axes[0,0].set_ylabel('Heatmap', fontsize=13, fontweight='bold')
    axes[1,0].set_ylabel('Overlay',  fontsize=13, fontweight='bold')
    fig.suptitle('ResNet18 Grad-CAM: 不同深度卷积层的关注区域对比', fontsize# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[1,j].axis('off')

    axes[0,0].set_ylabel('Heatmap', fontsize=13, fontweight='bold')
    axes[1,0].set_ylabel('Overlay',  fontsize=13, fontweight='bold')
    fig.suptitle('ResNet18 Grad-CAM: 不同深度卷积层的关注区域对比', fontsize=15, fontweight='bold', y=1.01)

    path = OUTPUT_DIR /# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[1,j].axis('off')

    axes[0,0].set_ylabel('Heatmap', fontsize=13, fontweight='bold')
    axes[1,0].set_ylabel('Overlay',  fontsize=13, fontweight='bold')
    fig.suptitle('ResNet18 Grad-CAM: 不同深度卷积层的关注区域对比', fontsize=15, fontweight='bold', y=1.01)

    path = OUTPUT_DIR / "gradcam_multi_layer.png"
    plt.tight_layout()
    plt.savefig(path,# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[1,j].axis('off')

    axes[0,0].set_ylabel('Heatmap', fontsize=13, fontweight='bold')
    axes[1,0].set_ylabel('Overlay',  fontsize=13, fontweight='bold')
    fig.suptitle('ResNet18 Grad-CAM: 不同深度卷积层的关注区域对比', fontsize=15, fontweight='bold', y=1.01)

    path = OUTPUT_DIR / "gradcam_multi_layer.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] Saved: {path}# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[1,j].axis('off')

    axes[0,0].set_ylabel('Heatmap', fontsize=13, fontweight='bold')
    axes[1,0].set_ylabel('Overlay',  fontsize=13, fontweight='bold')
    fig.suptitle('ResNet18 Grad-CAM: 不同深度卷积层的关注区域对比', fontsize=15, fontweight='bold', y=1.01)

    path = OUTPUT_DIR / "gradcam_multi_layer.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] Saved: {path}")

    # 单样本叠加图
    print("\n  生成单样本叠加对比图# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[1,j].axis('off')

    axes[0,0].set_ylabel('Heatmap', fontsize=13, fontweight='bold')
    axes[1,0].set_ylabel('Overlay',  fontsize=13, fontweight='bold')
    fig.suptitle('ResNet18 Grad-CAM: 不同深度卷积层的关注区域对比', fontsize=15, fontweight='bold', y=1.01)

    path = OUTPUT_DIR / "gradcam_multi_layer.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] Saved: {path}")

    # 单样本叠加图
    print("\n  生成单样本叠加对比图...")
    hl = model.layer4.layers[1].conv2  # 最深层的# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[1,j].axis('off')

    axes[0,0].set_ylabel('Heatmap', fontsize=13, fontweight='bold')
    axes[1,0].set_ylabel('Overlay',  fontsize=13, fontweight='bold')
    fig.suptitle('ResNet18 Grad-CAM: 不同深度卷积层的关注区域对比', fontsize=15, fontweight='bold', y=1.01)

    path = OUTPUT_DIR / "gradcam_multi_layer.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] Saved: {path}")

    # 单样本叠加图
    print("\n  生成单样本叠加对比图...")
    hl = model.layer4.layers[1].conv2  # 最深层的 conv
    hm = generate_heatmap_reg# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[1,j].axis('off')

    axes[0,0].set_ylabel('Heatmap', fontsize=13, fontweight='bold')
    axes[1,0].set_ylabel('Overlay',  fontsize=13, fontweight='bold')
    fig.suptitle('ResNet18 Grad-CAM: 不同深度卷积层的关注区域对比', fontsize=15, fontweight='bold', y=1.01)

    path = OUTPUT_DIR / "gradcam_multi_layer.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] Saved: {path}")

    # 单样本叠加图
    print("\n  生成单样本叠加对比图...")
    hl = model.layer4.layers[1].conv2  # 最深层的 conv
    hm = generate_heatmap_regression(model, hl, x)
    fig2, axs = plt.subplots(1,# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[1,j].axis('off')

    axes[0,0].set_ylabel('Heatmap', fontsize=13, fontweight='bold')
    axes[1,0].set_ylabel('Overlay',  fontsize=13, fontweight='bold')
    fig.suptitle('ResNet18 Grad-CAM: 不同深度卷积层的关注区域对比', fontsize=15, fontweight='bold', y=1.01)

    path = OUTPUT_DIR / "gradcam_multi_layer.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] Saved: {path}")

    # 单样本叠加图
    print("\n  生成单样本叠加对比图...")
    hl = model.layer4.layers[1].conv2  # 最深层的 conv
    hm = generate_heatmap_regression(model, hl, x)
    fig2, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].imshow(org); axs[0].set_title('Original Image\n(DonkeyCar 第一视角# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[1,j].axis('off')

    axes[0,0].set_ylabel('Heatmap', fontsize=13, fontweight='bold')
    axes[1,0].set_ylabel('Overlay',  fontsize=13, fontweight='bold')
    fig.suptitle('ResNet18 Grad-CAM: 不同深度卷积层的关注区域对比', fontsize=15, fontweight='bold', y=1.01)

    path = OUTPUT_DIR / "gradcam_multi_layer.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] Saved: {path}")

    # 单样本叠加图
    print("\n  生成单样本叠加对比图...")
    hl = model.layer4.layers[1].conv2  # 最深层的 conv
    hm = generate_heatmap_regression(model, hl, x)
    fig2, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].imshow(org); axs[0].set_title('Original Image\n(DonkeyCar 第一视角)', fontsize=12); axs[0].axis('off')
    im2 = ax# -*- coding: utf-8 -*-
"""
demo_gradcam.py - ResNet18 Grad-CAM 可视化演示 (DonkeyCar 数据集)

输出:
  demos/output/gradcam_multi_layer.png  - 不同深度层的 Grad-CAM 对比
  demos/output/gradcam_overlay.png      - 单样本热力图叠加
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'code'))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tests' / 'test_donkey'))

import os, numpy as np, cv2, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from eneuro.base import Tensor
from eneuro.explainability import GradCAM, get_all_conv_layers
from eneuro.utils.serializer import Serializer
from eneuro.utils.hooks import HookRegistry, capture_features, capture_gradients
from eneuro.nn.module import ResNet18AutoDrive

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def load_model():
    model_path = Path(__file__).resolve().parent.parent / "tests" / "test_donkey" / "results" / "model_200.json"
    model = ResNet18AutoDrive()
    Serializer().load(model, str(model_path))
    model = model.to('cpu')
    return model


def generate_heatmap_regression(model, target_layer, input_tensor):
    """回归任务的 Grad-CAM 热力图生成（复用已有逻辑）"""
    fh, fs = capture_features(target_layer)
    registry = HookRegistry(); registry.start_recording_sequence()
    output = model(input_tensor)
    registry.stop_recording_sequence()

    act = fs['output']
    grad_layer = registry.get_successor_layer(target_layer)
    if grad_layer is None:
        grad_layer = target_layer
    gh, gs = capture_gradients(grad_layer)
    output[0,0].backward()
    acts, grads = act[0], gs['grad_output'][0]

    if grads.ndim == 3:
        alpha = grads.mean(axis=(1,2))
    elif grads.ndim == 2:
        alpha = grads.mean(axis=1)
    else:
        alpha = grads
    hm = np.maximum(0.0, np.sum(alpha[:,None,None]*acts, axis=0))
    hm = hm / (np.max(hm) + 1e-8)

    fh.remove(); gh.remove()
    return hm


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("  ResNet18 Grad-CAM 可视化演示 (DonkeyCar 回归任务)")
    print("=" * 65)

    # 加载模型
    model = load_model()
    print("\n[1/3] 模型加载完成: ResNet18AutoDrive")

    # 加载验证集数据
    from dataset import AutoDriveDataset, preprocess_image
    ds = AutoDriveDataset(mode="val", transform=preprocess_image)
    print(f"[2/3] 数据集加载完成: {len(ds)} 张验证图像")

    # 选取5个不同深度的卷积层
    print("[3/3] 生成多深度层 Grad-CAM 对比图...\n")

    layers = {
        'conv1\n(shallow)':         model.conv1,
        'l1.block2.conv2\n(low)':   model.layer1.layers[1].conv2,
        'l2.block2.conv2\n(mid-low)': model.layer2.layers[1].conv2,
        'l3.block2.conv2\n(mid-high)':model.layer3.layers[1].conv2,
        'l4b.conv2\n(deepest)':     model.layer4.layers[1].conv2,
    }

    img, lbl = ds[15]
    x = Tensor(img[np.newaxis, ...])
    org = np.clip(img.transpose(1,2,0), 0, 1)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    names = list(layers.keys())
    for j, name in enumerate(names):
        hm = generate_heatmap_regression(model, layers[name], x)
        print(f"  [{name.replace(chr(10),' ')}] 热力图 shape={hm.shape}")

        im = axes[0,j].imshow(hm, cmap='jet', interpolation='bilinear')
        axes[0,j].set_title(name.replace('\n','\n'), fontsize=10, fontweight='bold')
        axes[0,j].axis('off')
        plt.colorbar(im, ax=axes[0,j], fraction=0.046)

        z = (org.shape[0]/hm.shape[0], org.shape[1]/hm.shape[1])
        hm_colored = np.array(plt.cm.jet(zoom(hm, z, order=1)))[:,:,:3]
        axes[1,j].imshow(0.5*org + 0.5*hm_colored)
        axes[1,j].set_title('Overlay', fontsize=9)
        axes[1,j].axis('off')

    axes[0,0].set_ylabel('Heatmap', fontsize=13, fontweight='bold')
    axes[1,0].set_ylabel('Overlay',  fontsize=13, fontweight='bold')
    fig.suptitle('ResNet18 Grad-CAM: 不同深度卷积层的关注区域对比', fontsize=15, fontweight='bold', y=1.01)

    path = OUTPUT_DIR / "gradcam_multi_layer.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  [OK] Saved: {path}")

    # 单样本叠加图
    print("\n  生成单样本叠加对比图...")
    hl = model.layer4.layers[1].conv2  # 最深层的 conv
    hm = generate_heatmap_regression(model, hl, x)
    fig2, axs = plt.subplots(1, 3, figsize=(16, 5))
    axs[0].imshow(org); axs[0].set_title('Original Image\n(DonkeyCar 第一视角)', fontsize=12); axs[0].axis('off')
    im2 = axs[1].imshow(hm, cmap='jet'); axs[1].set_title(f'Grad-CAM Heatmap\n