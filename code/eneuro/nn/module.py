import os
import weakref
import numpy as np
from ..base import functions as F
from ..base import *
from ..base.parameter import Parameter
from ..base.functions import pair
from ..base.functions import depthwise_conv2d, grouped_conv2d
from ..utils.statedict import StateDict

from ..global_config import VISUAL_CONFIG
import cv2

#function为方法，通用工具舍弃保存参数功能，layer为层，保存参数功能，也具备层层嵌套功能
class Layer:
    def __init__(self):
        self._params = set()
    #重写setattr方法，当设置的属性是Parameter或Layer时，将属性名加入_params集合，自动识别参数和子层
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]

        if len(outputs) == 0:
            raise
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()
    
    #参数生成器，递归获取所有参数
    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

    def get_params_list(self) -> list['Parameter']:
        """
        收集并返回当前层及其所有子层中的所有可训练参数。

        Returns:
            list[Parameter]: 一个包含所有 Parameter 对象的列表。
        """
        params_dict = {}
        self._flatten_params(params_dict)
        # 从字典的值中创建一个列表并返回
        return list(params_dict.values())

    #为保存参数来记录参数字典
    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

#在function类中已设置常用的函数支持反向传播，后续layer只需要前向传播自动形成计算图，支持反向传播
#layer类中已设置参数管理功能，后续layer只需要定义参数即可自动支持参数管理

class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        #初始化权重，如果指定了输入尺寸则初始化，否则在前向传播时根据输入数据动态初始化
        if self.in_size is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, inputs):
        if self.W.data is None:
            self.in_size = inputs.shape[1]
            self._init_W()

        y = linear(inputs, self.W, self.b)
        return y
    
class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None, visualize=False, 
                 groups=1, depthwise=False, dilation=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype
        self.visualize = visualize
        self.groups = groups
        self.depthwise = depthwise
        self.dilation = dilation

        # 深度可分离卷积的特殊处理
        if self.depthwise:
            if self.in_channels is None:
                raise ValueError("depthwise convolution requires in_channels to be specified")
            self.groups = self.in_channels
            assert out_channels % in_channels == 0, "out_channels must be divisible by in_channels for depthwise convolution"
            self.channel_multiplier = out_channels // in_channels
        else:
            assert out_channels % groups == 0, "out_channels must be divisible by groups"
            if in_channels is not None:
                assert in_channels % groups == 0, "in_channels must be divisible by groups"

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        
        if self.depthwise:
            # 深度可分离卷积的权重形状: (OC, 1, KH, KW)
            scale = np.sqrt(1 / (C * KH * KW))
            W_data = xp.random.randn(OC, 1, KH, KW).astype(self.dtype) * scale
        else:
            # 普通卷积或分组卷积的权重形状: (OC, C//groups, KH, KW)
            scale = np.sqrt(1 / ((C // self.groups) * KH * KW))
            W_data = xp.random.randn(OC, C // self.groups, KH, KW).astype(self.dtype) * scale
        
        self.W.data = W_data

    def forward(self, inputs):
        if self.W.data is None:
            self.in_channels = inputs.shape[1]          
            self._init_W()

        if self.depthwise:
            # 深度可分离卷积: 先进行逐通道卷积，再进行1x1卷积
            # 逐通道卷积
            y = depthwise_conv2d(
                inputs, self.W, None,
                stride=self.stride,
                pad=self.pad,
                dilation=self.dilation,
                visualize=self.visualize
            )
            # 1x1卷积
            if self.channel_multiplier > 1:
                # 创建1x1卷积核
                C = self.in_channels
                OC = self.out_channels
                w_shape = (OC, C, 1, 1)
                scale = np.sqrt(1 / C)
                w_data = np.random.randn(*w_shape).astype(self.dtype) * scale
                w = Parameter(w_data, name='W_1x1')
                y = conv2d(y, w, self.b, stride=1, pad=0, visualize=self.visualize)
            else:
                if self.b is not None:
                    y += self.b.reshape(1, -1, 1, 1)
        elif self.groups > 1:
            # 分组卷积
            y = grouped_conv2d(
                inputs, self.W, self.b,
                stride=self.stride,
                pad=self.pad,
                groups=self.groups,
                dilation=self.dilation,
                visualize=self.visualize
            )
        else:
            # 普通卷积
            y = conv2d(
                inputs, self.W, self.b,
                stride=self.stride,
                pad=self.pad,
                dilation=self.dilation,
                visualize=self.visualize
            )
        return y
# 转置卷积层
class Deconv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None, visualize=False):
  
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype
        self.visualize = visualize

        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

    def _init_W(self, xp=np):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        W_data = xp.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            self._init_W()

        y = deconv2d(x, self.W, self.b, self.stride, self.pad, visualize=self.visualize)
        return y

from ..base import Config

class BatchNorm(Layer):
    def __init__(self, num_features, num_dims=4, eps=1e-5, momentum=0.9, dtype=np.float32):
        super().__init__()
        self.num_features = num_features
        self.num_dims = num_dims
        self.eps = eps
        self.momentum = momentum
        self.dtype = dtype

        # 为兼容 BatchNorm2d 后端，gamma/beta 统一使用 (C,) 形状
        self.gamma = Parameter(np.ones(num_features, dtype=dtype), name='gamma')
        self.beta = Parameter(np.zeros(num_features, dtype=dtype), name='beta')

        # 运行统计量以 Parameter 形式保存（不参与梯度）
        self.running_mean = Parameter(np.zeros(num_features, dtype=dtype), name='running_mean')
        self.running_mean.requires_grad = False
        self.running_var = Parameter(np.ones(num_features, dtype=dtype), name='running_var')
        self.running_var.requires_grad = False

        # 兼容旧属性名
        self.moving_mean = self.running_mean.data
        self.moving_var = self.running_var.data

    def forward(self, x):
        if self.num_dims == 2:
            n, c = x.shape
            x_4d = x.reshape(n, c, 1, 1)
            y_4d = batch_norm2d((x_4d, self.gamma, self.beta), self.running_mean, self.running_var, self.momentum, self.eps)
            y = y_4d.reshape(n, c)
        else:
            y = batch_norm2d((x, self.gamma, self.beta), self.running_mean, self.running_var, self.momentum, self.eps)

        # 保持旧字段同步
        self.moving_mean = self.running_mean.data
        self.moving_var = self.running_var.data
        return y

class BatchNorm2d(Layer):
    def __init__(self, out_channels, momentum=0.9, eps=1e-5, dtype=np.float32):
        super().__init__()
        self.out_channels = out_channels
        self.momentum = momentum
        self.eps = eps

        # 可训练参数 γ 和 β
        self.gamma = Parameter(np.ones(out_channels, dtype=dtype), name='gamma')
        self.beta  = Parameter(np.zeros(out_channels, dtype=dtype), name='beta')
        # 不可训练的运行统计量（仍用 Tensor 存储，但 requires_grad=False）
        self.running_mean = Parameter(np.zeros(out_channels, dtype=dtype), name='running_mean')
        self.running_mean.requires_grad=False
        self.running_var  = Parameter(np.ones(out_channels, dtype=dtype), name='running_var')
        self.running_var.requires_grad=False

    def forward(self, inputs):
        x = (inputs, self.gamma, self.beta)
        y = batch_norm2d(x, self.running_mean, self.running_var, self.momentum, self.eps)
        return y
    
class FusedConvReLU(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, 
                 nobias=False, dtype=np.float32, in_channels=None, visualize=False, 
                 groups=1, depthwise=False, dilation=1):
        super().__init__()
        self.conv = Conv2d(out_channels, kernel_size, stride, pad, 
                 nobias, dtype, in_channels, visualize, 
                 groups, depthwise, dilation)

    def forward(self, inputs):
        if self.conv.depthwise:
            # 深度可分离卷积: 先进行逐通道卷积，再进行1x1卷积
            print("Warning: Not supported fusion (depthwise conv). Using not fused function.")
            y = self.conv.forward(inputs)
            y = relu(y)

        elif self.conv.groups > 1:
            # 分组卷积
            print("Warning: Not supported fusion (grouped conv). Using not fused function.")
            y = self.conv.forward(inputs)
            y = relu(y)
            
        else:
            # 普通卷积
            y = fused_conv_relu(inputs, self.conv.W, self.conv.b, self.conv.stride, 
                                self.conv.pad, self.conv.dilation, self.conv.visualize)
        return y

class FusedConvBNReLU(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, 
                 nobias=False, dtype=np.float32, in_channels=None, visualize=False, 
                 groups=1, depthwise=False, dilation=1,
                 momentum=0.9, eps=1e-5):
        super().__init__()
        self.conv = Conv2d(out_channels, kernel_size, stride, pad, 
                 nobias, dtype, in_channels, visualize, 
                 groups, depthwise, dilation)
        self.bn = BatchNorm2d(out_channels, momentum, eps)
        self.visualize = visualize

    def forward(self, inputs):
        if self.conv.depthwise:
            # 深度可分离卷积: 先进行逐通道卷积，再进行1x1卷积
            print("Warning: Not supported fusion (depthwise conv). Using not fused function.")
            y = self.conv.forward(inputs)
            y = self.bn.forward(y)
            y = relu(y)

        elif self.conv.groups > 1:
            # 分组卷积
            print("Warning: Not supported fusion (grouped conv). Using not fused function.")
            y = self.conv.forward(inputs)
            y = self.bn.forward(y)
            y = relu(y)
            
        else:
            # 普通卷积
            y = fused_conv_bn_relu(
                inputs, self.conv.W, self.conv.b, 
                self.bn.gamma, self.bn.beta,
                self.bn.running_mean, self.bn.running_var,
                stride=self.conv.stride, pad=self.conv.pad, dilation=self.conv.dilation,
                momentum=self.bn.momentum, eps=self.bn.eps,
                visualize=self.visualize
            )
        return y

#由于池化层，relu函数等不需要参数，这些可以直接用function中的函数即可

#用于连接层和加载参数的类class Module(Layer):
class Module(Layer,StateDict):
    def __init__(self):
        Layer.__init__(self)
        # 添加模型特有的属性
        self.metadata = {
            'model_class': self.__class__.__name__,
            'version': '1.0'
        }

    def to_dict(self):
        """序列化：所有参数都是Tensor，直接处理"""
        params_dict = {}
        self._flatten_params(params_dict)  # 输出全是Tensor类型

        serializable_params = {}
        for key, param in params_dict.items():
            # 无需类型判断：param一定是Tensor
            serializable_params[key] = {
                'data': self._to_pure_list(param.data),
                'grad': self._to_pure_list(param.grad.data) if (
                            param.grad is not None and hasattr(param.grad, 'data')) else None,
                'requires_grad': param.requires_grad,
                'name': param.name
            }

        return {
            'metadata': self.metadata,
            'params': serializable_params,
            'training': getattr(self, 'training', True),
            'model_class': self.__class__.__name__
        }

    def from_dict(self, d):
        """反序列化：所有参数都是Tensor，直接恢复"""
        params_data = d.get('params', {})
        current_params = {}
        self._flatten_params(current_params)  # 输出全是Tensor类型

        for key, param_data in params_data.items():
            if key in current_params:
                # 无需类型判断：current_params[key]一定是Tensor
                param = current_params[key]
                # 恢复Tensor核心属性
                param.data = np.array(self._from_pure_list(param_data['data']))
                # 恢复梯度（仍是Tensor类型）
                if 'grad' in param_data and param_data['grad'] is not None:
                    param.grad = Tensor(np.array(self._from_pure_list(param_data['grad'])), requires_grad=False)
                else:
                    param.grad = None
                # 恢复其他属性
                param.requires_grad = param_data.get('requires_grad', False)
                param.name = param_data.get('name')

        # 恢复训练状态
        if 'training' in d:
            self.training = d['training']

    # 辅助函数：仅处理numpy数组→纯列表（适配Tensor）
    def _to_pure_list(self, tensor_like):
        if isinstance(tensor_like, np.ndarray):
            return tensor_like.tolist()
        elif isinstance(tensor_like, (list, tuple)):
            return [self._to_pure_list(item) for item in tensor_like]
        elif isinstance(tensor_like, (int, float, bool)):
            return tensor_like
        else:
            return float(tensor_like)

    def _from_pure_list(self, pure_list):
        if isinstance(pure_list, list):
            return [self._from_pure_list(item) for item in pure_list]
        else:
            return pure_list
    
    #"""类似PyTorch的state_dict，只返回参数数据"""
    # def state_dict(self):    
    #     params_dict = {}
    #     self._flatten_params(params_dict)
        
    #     state = {}
    #     for key, param in params_dict.items():
    #         if hasattr(param, 'data'):
    #             state[key] = {
    #                 'data': param.data,
    #                 'requires_grad': getattr(param, 'requires_grad', True)
    #             }
    #     return state


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

#多层感知机
class MLP(Module):
    def __init__(self, fc_output_sizes, activation=sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, inputs):
        for l in self.layers[:-1]:
            inputs = self.activation(l(inputs))
        return self.layers[-1](inputs)


#简单的卷积模型
class CNNWithPooling(Module):
    def __init__(self, in_channels=3, num_classes=10):
      
        super().__init__()
        self.layers = []
        
        # 卷积层部分
        layers = [
            Conv2d(in_channels=in_channels,out_channels=32, kernel_size=3, pad=1),  # 卷积层
            relu,                   # 激活函数
            F.Pooling(kernel_size=2,stride=2),        # 池化层
            
            Conv2d(out_channels=64, kernel_size=3, pad=1),
            relu,
            F.Pooling(kernel_size=2,stride=2),
       
        ]
        
        # 全连接层部分
        fc_layers = [
            flatten,
            Linear(128),
            relu,
            Linear(64),
            sigmoid,                # Sigmoid激活
            Linear(num_classes),
            softmax              # Softmax输出
        ]
        
        # 合并所有层
        all_layers = layers + fc_layers
        
        # 动态设置属性
        for i, layer in enumerate(all_layers):
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super().__init__()
        self.conv1 = Conv2d(out_channels, kernel_size=3, stride=stride, pad=1)
        self.bn1 = BatchNorm(out_channels)
        self.relu = relu
        self.conv2 = Conv2d(out_channels, kernel_size=3, stride=1, pad=1)
        self.bn2 = BatchNorm(out_channels)
        if downsample:
            self.conv3 = Conv2d(out_channels, kernel_size=1, stride=stride, in_channels=in_channels)
        else:
            self.conv3 = None
        self.downsample = downsample
    
    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.relu(x + y)
        return y