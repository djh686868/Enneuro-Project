import os
import weakref
import numpy as np
from ..base import functions as F
from ..base import *
from ..base.parameter import Parameter
from ..base.functions import pair
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
                 pad=0, nobias=False, dtype=np.float32, in_channels=None,visualize=False):
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
        W_data = xp.random.randn(OC, C, KH, KW).astype(self.dtype) * scale
        self.W.data = W_data

    def forward(self, inputs):
        if self.W.data is None:
            self.in_channels = inputs.shape[1]          
            self._init_W()

        y = conv2d(inputs, self.W, self.b, self.stride, self.pad,self.visualize)
        return y
#反卷积层    一般用不到
# class Deconv2d(Layer):
#     def __init__(self, out_channels, kernel_size, stride=1,
#                  pad=0, nobias=False, dtype=np.float32, in_channels=None):
  
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.pad = pad
#         self.dtype = dtype

#         self.W = Parameter(None, name='W')
#         if in_channels is not None:
#             self._init_W()

#         if nobias:
#             self.b = None
#         else:
#             self.b = Parameter(np.zeros(out_channels, dtype=dtype), name='b')

#     def _init_W(self, xp=np):
#         C, OC = self.in_channels, self.out_channels
#         KH, KW = pair(self.kernel_size)
#         scale = np.sqrt(1 / (C * KH * KW))
#         W_data = xp.random.randn(C, OC, KH, KW).astype(self.dtype) * scale
#         self.W.data = W_data

#     def forward(self, x):
#         if self.W.data is None:
#             self.in_channels = x.shape[1]
#             self._init_W()

#         y = F.deconv2d(x, self.W, self.b, self.stride, self.pad)
#         return y

# module.py 中添加

class BatchNorm2d(Layer):
    def __init__(self, out_channels, momentum=0.9, eps=1e-5):
        super().__init__()
        self.out_channels = out_channels
        self.momentum = momentum
        self.eps = eps

        # 可训练参数 γ 和 β
        self.gamma = Parameter(np.ones(out_channels, dtype=np.float32), name='gamma')
        self.beta  = Parameter(np.zeros(out_channels, dtype=np.float32), name='beta')
        # 不可训练的运行统计量（仍用 Tensor 存储，但 requires_grad=False）
        self.running_mean = Tensor(np.zeros(out_channels, dtype=np.float32), requires_grad=False, name='running_mean')
        self.running_var  = Tensor(np.ones(out_channels, dtype=np.float32), requires_grad=False, name='running_var')

    def forward(self, inputs):
        x = (inputs, self.gamma, self.beta)
        y = batch_norm2d(x, self.running_mean, self.running_var, self.momentum, self.eps)
        return y
    
class FusedConvReLU(Conv2d):
    def forward(self, inputs):
        if self.W.data is None:
            self.in_channels = inputs.shape[1]          
            self._init_W()

        y = fused_conv_relu(inputs, self.W, self.b, self.stride, self.pad,self.visualize)
        return y

class FusedConvBNReLU(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0,
                 momentum=0.9, eps=1e-5, in_channels=None, visualize=False):
        super().__init__()
        self.conv = Conv2d(out_channels, kernel_size, stride, pad, 
                           nobias=True, in_channels=in_channels, visualize=visualize)
        self.bn = BatchNorm2d(out_channels, momentum, eps)
        self.visualize = visualize

    def forward(self, inputs):
        # 手动调用融合函数
        return fused_conv_bn_relu(
            inputs, self.conv.W, self.conv.b, 
            self.bn.gamma, self.bn.beta,
            self.bn.running_mean, self.bn.running_var,
            stride=self.conv.stride, pad=self.conv.pad,
            momentum=self.bn.momentum, eps=self.bn.eps,
            visualize=self.visualize
        )

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
