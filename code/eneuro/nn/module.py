import os
import weakref
import numpy as np
from core.functions import Function as F
from core.parameter import Parameter
from core.functions import pair



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

    def forward(self, x):
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y
    
class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1,
                 pad=0, nobias=False, dtype=np.float32, in_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.dtype = dtype

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

    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]          
            self._init_W()

        y = F.conv2d(x, self.W, self.b, self.stride, self.pad)
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



#由于池化层，relu函数等不需要参数，这些可以直接用function中的函数即可

#用于连接层和加载参数的类class Module(Layer):
class Module(Layer):
    def __init__(self):
        super().__init__()
        # 添加模型特有的属性
        self.metadata = {
            'model_class': self.__class__.__name__,
            'version': '1.0'
        }
    
    def to_dict(self):
        """实现 Module 接口：完整序列化模型"""
        # 使用 _flatten_params 获取所有参数
        params_dict = {}
        self._flatten_params(params_dict)
        
        # 转换为可序列化的格式
        serializable_params = {}
        for key, param in params_dict.items():
            if hasattr(param, 'data'):
                # 假设 Parameter 有 data 和 grad 属性
                serializable_params[key] = {
                    'data': param.data.tolist() if hasattr(param.data, 'tolist') else param.data,
                    'grad': param.grad.tolist() if param.grad is not None and hasattr(param.grad, 'tolist') else param.grad,
                    'requires_grad': getattr(param, 'requires_grad', True)
                }
        
        return {
            'metadata': self.metadata,
            'params': serializable_params,
            'training': getattr(self, 'training', True),
            'model_class': self.__class__.__name__
        }
    
    def from_dict(self, data):
        """实现 Module 接口：从字典恢复模型"""
        params_data = data.get('params', {})
        
        # 获取当前模型的扁平化参数
        current_params = {}
        self._flatten_params(current_params)
        
        # 恢复参数值
        for key, param_data in params_data.items():
            if key in current_params:
                param = current_params[key]
                if 'data' in param_data and hasattr(param, 'data'):
                    # 恢复数据
                    param.data = np.array(param_data['data']) if isinstance(param_data['data'], list) else param_data['data']
                
                if 'grad' in param_data and param_data['grad'] is not None:
                    # 恢复梯度
                    param.grad = np.array(param_data['grad']) if isinstance(param_data['grad'], list) else param_data['grad']
                
                if 'requires_grad' in param_data:
                    setattr(param, 'requires_grad', param_data['requires_grad'])
        
        # 恢复训练状态
        if 'training' in data:
            self.training = data['training']
    
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

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

#多层感知机
class MLP(Module):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)


#简单的卷积模型
class CNNWithPooling(Module):
    def __init__(self, in_channels=3, num_classes=10):
      
        super().__init__()
        self.layers = []
        
        # 卷积层部分
        layers = [
            Conv2d(32, 3, pad=1),  # 卷积层
            F.relu,                   # 激活函数
            F.Pooling(2, 2),        # 池化层
            
            Conv2d(64, 3, pad=1),
            F.relu,
            F.Pooling(2, 2),
       
        ]
        
        # 全连接层部分
        fc_layers = [
            Linear(128),
            F.relu,
            Linear(64),
            F.sigmoid,                # Sigmoid激活
            Linear(num_classes),
            F.softmax              # Softmax输出
        ]
        
        # 合并所有层
        all_layers = layers + fc_layers
        
        # 动态设置属性
        for i, layer in enumerate(all_layers):
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x