# 卷积模块核心代码伪代码

## 1. 普通卷积 (Conv2d)

```python
class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, 
                 nobias=False, in_channels=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        
        self.W = Parameter(None, name='W')
        if in_channels is not None:
            self._init_W()
        
        self.b = None if nobias else Parameter(np.zeros(out_channels), name='b')
    
    def _init_W(self):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        self.W.data = np.random.randn(OC, C, KH, KW) * scale
    
    def forward(self, inputs):
        if self.W.data is None:
            self.in_channels = inputs.shape[1]
            self._init_W()
        
        return conv2d(inputs, self.W, self.b, self.stride, self.pad)
```

## 2. 分组卷积 (Grouped Convolution)

```python
class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, 
                 nobias=False, in_channels=None, groups=1):
        # 与普通卷积相同的部分
        ...
        
        self.groups = groups
        assert out_channels % groups == 0
        assert in_channels % groups == 0
        
        # 与普通卷积相同的部分
        ...
    
    def _init_W(self):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / ((C // self.groups) * KH * KW))
        self.W.data = np.random.randn(OC, C//self.groups, KH, KW) * scale
    
    def forward(self, inputs):
        # 与普通卷积相同的部分
        ...
        
        return grouped_conv2d(inputs, self.W, self.b, self.stride, self.pad, self.groups)
```

## 3. 深度可分离卷积 (Depthwise Separable Convolution)

```python
class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, 
                 nobias=False, in_channels=None, depthwise=False):
        # 与普通卷积相同的部分
        ...
        
        self.depthwise = depthwise
        if depthwise:
            self.groups = in_channels
            assert out_channels % in_channels == 0
            self.channel_multiplier = out_channels // in_channels
        
        # 与普通卷积相同的部分
        ...
    
    def _init_W(self):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        self.W.data = np.random.randn(OC, 1, KH, KW) * scale
    
    def forward(self, inputs):
        # 与普通卷积相同的部分
        ...
        
        # 逐通道卷积
        y = depthwise_conv2d(inputs, self.W, None, self.stride, self.pad)
        
        # 1x1卷积
        if self.channel_multiplier > 1:
            C, OC = self.in_channels, self.out_channels
            w = Parameter(np.random.randn(OC, C, 1, 1) * np.sqrt(1/C), name='W_1x1')
            y = conv2d(y, w, self.b, 1, 0)
        elif self.b is not None:
            y += self.b.reshape(1, -1, 1, 1)
        
        return y
```

## 4. 扩张卷积 (Dilated Convolution)

```python
class Conv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, 
                 nobias=False, in_channels=None, dilation=1):
        # 与普通卷积相同的部分
        ...
        
        self.dilation = dilation  # 扩张率，控制空洞大小
        
        # 与普通卷积相同的部分
        ...
    
    def _init_W(self):
        # 与普通卷积相同的部分
        ...
    
    def forward(self, inputs):
        # 与普通卷积相同的部分
        ...
        
        # 执行扩张卷积，通过dilation参数控制空洞大小
        return conv2d(inputs, self.W, self.b, self.stride, self.pad, self.dilation)
```

### 扩张卷积的空洞特性

扩张卷积通过在卷积核中插入空洞（零值）来增加感受野，而不增加参数量。例如：

- 当 dilation=1 时，3×3 卷积核为：
  ```
  [1 1 1]
  [1 1 1]
  [1 1 1]
  ```

- 当 dilation=2 时，3×3 卷积核变为：
  ```
  [1 0 1 0 1]
  [0 0 0 0 0]
  [1 0 1 0 1]
  [0 0 0 0 0]
  [1 0 1 0 1]
  ```
  （实际计算时通过跳过采样实现，而非物理填充零值）

- 当 dilation=3 时，感受野进一步增大。

## 5. 转置卷积 (Transposed Convolution)

```python
class Deconv2d(Layer):
    def __init__(self, out_channels, kernel_size, stride=1, pad=0, 
                 nobias=False, in_channels=None):
        # 与普通卷积相似的部分
        ...
    
    def _init_W(self):
        C, OC = self.in_channels, self.out_channels
        KH, KW = pair(self.kernel_size)
        scale = np.sqrt(1 / (C * KH * KW))
        self.W.data = np.random.randn(C, OC, KH, KW) * scale
    
    def forward(self, x):
        if self.W.data is None:
            self.in_channels = x.shape[1]
            self._init_W()
        
        return deconv2d(x, self.W, self.b, self.stride, self.pad)
```

## 6. 核心卷积函数

### 6.1 conv2d 函数

```python
def conv2d(x, W, b=None, stride=(1,1), pad=(0,0), dilation=1):
    # 1. 输入转列
    col = im2col_array(x, W.shape[2:], stride, pad, dilation=dilation)
    # 2. 张量点积
    y = np.tensordot(col, W, ((1, 2, 3), (1, 2, 3)))
    # 3. 添加偏置
    if b is not None:
        y += b
    # 4. 调整形状
    return np.rollaxis(y, 3, 1)
```

### 6.2 depthwise_conv2d 函数

```python
def depthwise_conv2d(x, W, b=None, stride=(1,1), pad=(0,0), dilation=1):
    N, C, H, W_in = x.shape
    OC, _, KH, KW = W.shape
    y = np.zeros((N, OC, 
                 get_conv_outsize(H, KH, stride[0], pad[0], dilation[0]), 
                 get_conv_outsize(W_in, KW, stride[1], pad[1], dilation[1])), 
                dtype=x.dtype)
    
    # 逐通道卷积
    for i in range(C):
        x_channel = x[:, i:i+1, :, :]
        W_channel = W[i*OC//C:(i+1)*OC//C, :, :, :]
        y[:, i*OC//C:(i+1)*OC//C, :, :] = conv2d(x_channel, W_channel, None, 
                                                  stride, pad, dilation)
    
    if b is not None:
        y += b.reshape(1, -1, 1, 1)
    
    return y
```

### 6.3 grouped_conv2d 函数

```python
def grouped_conv2d(x, W, b=None, stride=(1,1), pad=(0,0), groups=1, dilation=1):
    N, C, H, W_in = x.shape
    OC, C_per_group, KH, KW = W.shape
    OC_per_group = OC // groups
    y = np.zeros((N, OC, 
                 get_conv_outsize(H, KH, stride[0], pad[0], dilation[0]), 
                 get_conv_outsize(W_in, KW, stride[1], pad[1], dilation[1])), 
                dtype=x.dtype)
    
    # 分组卷积
    for i in range(groups):
        x_group = x[:, i*C_per_group:(i+1)*C_per_group, :, :]
        W_group = W[i*OC_per_group:(i+1)*OC_per_group, :, :, :]
        y[:, i*OC_per_group:(i+1)*OC_per_group, :, :] = conv2d(x_group, W_group, None, 
                                                              stride, pad, dilation)
    
    if b is not None:
        y += b.reshape(1, -1, 1, 1)
    
    return y
```

### 6.4 deconv2d 函数

```python
def deconv2d(x, W, b=None, stride=(1,1), pad=(0,0), outsize=None):
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    C, OC, KH, KW = W.shape
    N, C, H, W_in = x.shape
    
    # 计算输出尺寸
    if outsize is None:
        out_h = get_deconv_outsize(H, KH, SH, PH)
        out_w = get_deconv_outsize(W_in, KW, SW, PW)
    else:
        out_h, out_w = pair(outsize)
    
    # 执行转置卷积
    gcol = np.tensordot(W, x, (0, 1))
    gcol = np.rollaxis(gcol, 3)
    y = col2im_array(gcol, (N, OC, out_h, out_w), (KH, KW), stride, pad)
    
    if b is not None:
        y += b.reshape((1, b.size, 1, 1))
    
    return y
```

## 7. 辅助函数

### 7.1 get_conv_outsize 函数

```python
def get_conv_outsize(input_size, kernel_size, stride, pad, dilation=1):
    return (input_size + pad * 2 - (kernel_size - 1) * dilation - 1) // stride + 1
```

### 7.2 im2col_array 函数

```python
def im2col_array(img, kernel_size, stride, pad, dilation=1):
    N, C, H, W = img.shape
    ...     
    DH, DW = pair(dilation)  # 扩张率，控制采样步长
    ...
    for j in range(KH):
        for i in range(KW):
            # j*DH 和 i*DW 控制起始位置
            # ::SH 和 ::SW 控制步长
            # 这样就实现了在输入上跳过采样，相当于在卷积核中插入空洞
            col[:, :, j, i, :, :] = img[:, :, j*DH::SH, i*DW::SW]
    
    return col
```

### 扩张卷积的空洞实现原理

扩张卷积通过在 **输入特征图** 上的非连续采样来实现空洞效果，而不是在 **卷积核** 中物理填充零值。具体来说：

1. **采样位置计算**：对于大小为 k 	imes k 的卷积核，当扩张率为 d 时，采样位置为 0, d, 2d, ..., (k-1)d。

2. **im2col 实现**：在 `im2col_array` 函数中，通过 `j*DH::SH` 和 `i*DW::SW` 的切片操作，实现了对输入特征图的跳过采样。

3. **感受野增大**：随着扩张率 d 的增加，卷积核在输入特征图上覆盖的区域增大，从而增加了感受野。

4. **参数量不变**：虽然感受野增大了，但卷积核的大小 k 	imes k 保持不变，因此参数量也不变。

### 示例：3×3 卷积核的不同扩张率

- **dilation=1**：
  - 采样位置：(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)
  - 感受野大小：3×3

- **dilation=2**：
  - 采样位置：(0,0), (0,2), (0,4), (2,0), (2,2), (2,4), (4,0), (4,2), (4,4)
  - 感受野大小：5×5

- **dilation=3**：
  - 采样位置：(0,0), (0,3), (0,6), (3,0), (3,3), (3,6), (6,0), (6,3), (6,6)
  - 感受野大小：7×7

通过这种方式，扩张卷积能够在不增加参数量的情况下，有效地增大感受野，这对于语义分割等需要大感受野的任务非常重要。

## 8. 支持的卷积核大小

- 1 × 1
- 3 × 3
- 5 × 5
- 11 × 11
- 31 × 31

## 9. 使用示例

### 9.1 普通卷积

```python
# 3x3 卷积
conv = Conv2d(out_channels=32, kernel_size=3, stride=1, pad=1, in_channels=3)
output = conv(input_tensor)
```

### 9.2 分组卷积

```python
# 3x3 分组卷积，2组
conv = Conv2d(out_channels=32, kernel_size=3, stride=1, pad=1, 
              in_channels=6, groups=2)
output = conv(input_tensor)
```

### 9.3 深度可分离卷积

```python
# 3x3 深度可分离卷积
conv = Conv2d(out_channels=64, kernel_size=3, stride=1, pad=1, 
              in_channels=32, depthwise=True)
output = conv(input_tensor)
```

### 9.4 扩张卷积

```python
# 3x3 扩张卷积，扩张率2
conv = Conv2d(out_channels=32, kernel_size=3, stride=1, pad=2, 
              in_channels=3, dilation=2)
output = conv(input_tensor)
```

### 9.5 转置卷积

```python
# 3x3 转置卷积
conv = Deconv2d(out_channels=3, kernel_size=3, stride=2, pad=1, 
                in_channels=32)
output = conv(input_tensor)
```