## 4-Bits 量化

### 1 大模型推理量化

#### 1.1 BaiChuan 模型量化

量化计算代码：

```python
class QLinear(torch.nn.Module):
    def __init__(self, bits: int, weight: torch.Tensor, bias=None):
        super().__init__()
        self.quant_bits = bits
        # 对称量化
        self.scale = weight.abs().max(dim=-1).values / ((2 ** (bits - 1)) - 1)
        self.scale = self.scale.to(torch.float32)
        if self.quant_bits == 4:
            self.weight = quant4(weight, self.scale)
            
def quant4(weight: torch.Tensor, scale: torch.Tensor):
   intweight = torch.clip(torch.round(weight.to(scale.dtype) / scale[:, None]), -16, 15).to(dtype=torch.int32)
```

#### 1.2 ChatGLM2 模型量化

```python
class QuantizedLinear(torch.nn.Module):
    def __init__(self, weight_bit_width: int, weight):
        self.weight_bit_width = weight_bit_width        
        self.weight_scale = weight.abs().max(dim=-1).values / ((2 ** (weight_bit_width - 1)) - 1)
        self.weight = torch.round(weight / self.weight_scale[:, None]).to(torch.int8)
        if weight_bit_width == 4:
            self.weight = compress_int4_weight(self.weight)
```

#### 1.3 对应量化计算公式

$$
scale = \frac{2·absmax}{2^b-2} =  \frac{absmax}{2^{(b-1)}-1} \\
weight = clip \bigg( round(weight / scale),-2^b,2^b-1\bigg) = round(weight / scale)
$$

由于  $weight / scale$  的取值范围在 $[-2^{(b-1)}+1,2^{(b-1)}-1$] 内，必定在$[-2^b,2^b-1]$ 范围内，因此不会发生截断的情况，也就不必使用 $clip$ 函数。

#### 1.4 BaiChuan 模型权重量化后存储转换

模型权重量化后得到整数索引，使用INT32数据表示，由此可以将8个INT4表示的整数索引合并成为1个INT32表示的整数。

```python
def quant4(weight: torch.Tensor, scale: torch.Tensor):
    stream = torch.cuda.current_stream()
    num_row = weight.size(0)
    num_chan_fp16 = weight.size(1)
    # 4bit
    num_chan_int = num_chan_fp16 // 8
    # 假设原来有64个元素，如果使用int32来表示int4(0-15)，则原来的每8个元素会使用一个int32表示，于是有计算num_chan_fp16 // 8
    qweight = torch.zeros((num_row, num_chan_int), dtype=torch.int32, device=weight.device)
    intweight = torch.clip(torch.round(weight.to(scale.dtype) / scale[:, None]), -16, 15).to(dtype=torch.int32)

    for j in range(num_chan_int): # 0x0f 表示 00001111，也就是15
        qweight[:, j] = ((intweight[:, j * 8 + 7] & 0x0f) << 28) \
                        | ((intweight[:, j * 8 + 6] & 0x0f) << 24) \
                        | ((intweight[:, j * 8 + 5] & 0x0f) << 20) \
                        | ((intweight[:, j * 8 + 4] & 0x0f) << 16) \
                        | ((intweight[:, j * 8 + 3] & 0x0f) << 12) \
                        | ((intweight[:, j * 8 + 2] & 0x0f) << 8) \
                        | ((intweight[:, j * 8 + 1] & 0x0f) << 4) \
                        | (intweight[:, j * 8] & 0x0f)
    return qweight
```

注解：

首先，说明一下 `((intweight[:, j * 8 + 1] & 0x0f) << 4)  ` 这个单元的含义，`intweight[:, j * 8 + 1]` 表示取第`j*8+1` 列所有元素的后四位，每个元素使用32位整数表示，`_ & 0x0f`表示**位与**运算，也就是 `_ & 00001111`， 实际就是取 `_` 的后4位表示，前28位在执行位与运算后都变为0。

其次， `<< ` 表示**位左移**运算，分别移动0/4/8/12/16/20/24/28位后变为：

```shell
[
a b c d 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 a b c d 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 a b c d 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 a b c d 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a b c d 0 0 0 0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a b c d 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a b c d 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 a b c d
]
```

再者，将以上8个数取**位或**运算后，将得到 `a b c d a b c d a b c d a b c d a b c d a b c d a b c d a b c d ` 表示，由此实现将8个int4表示合并成1个int32表示。

`((intweight[:, j * 8 + 1] & 0x0f) << 4) | (intweight[:, j * 8] & 0x0f)` ，代码含义是将一个二维数组`intweight`中的每个元素的二进制表示的后四位与前四位交换，然后将这两个四位二进制数合并成一个八位二进制数。这个操作是通过位运算符`&`和`|`实现的。具体来说，`(intweight[:, j * 8 + 1] & 0x0f)`表示取出`intweight`数组中第`j*8+1`列的所有元素的后四位，`(intweight[:, j * 8] & 0x0f)`表示取出第`j*8`列的所有元素的后四位。然后将这两个四位二进制数左移和右移，再用按位或运算符`|`合并成一个八位二进制数。



