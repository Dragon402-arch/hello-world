## Transformer

### 1. 参考文章 

#### 1.1 理论介绍

- [介绍1](https://jalammar.github.io/illustrated-transformer/)
- [介绍2](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0)

#### 1.2 代码实现

- [清爽seq2seq部分代码！！！](https://github.com/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb)
- [PyTorch官方代码](https://torchtutorialstaging.z5.web.core.windows.net/beginner/transformer_tutorial.html)
- [pre-norm代码](https://blog.floydhub.com/the-transformer-in-pytorch/)
- [wide方式注意力机制代码](https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51)
- [datawhale代码](https://mp.weixin.qq.com/s?__biz=MzIyNjM2MzQyNg==&mid=2247590454&idx=1&sn=5ad464171acf15d2c76e31c7205c42c5&chksm=e872b37bdf053a6d98d44f8d1ff5bbc4f20af9518e2033323d009563f27e560bec6985285f95&scene=126&sessionid=1631756715&key=dc182ef5baef827ddf6ad20e7d1066d6be9f1889ecc19ee247c25c44def7ab740fc6be9275501222cf39c05d735fa4849761264adb61b1d22bbd249e2a067a931b432bff0072a0fac536dcbd8676aca30fd86b9c4d7ecd472721851cf8c01bcfc3beecac4e33bf473d90a568e2624d4f58ce9f2aaf9edfa607ac1f958d9353f6&ascene=1&uin=MTU0NzYwMjkxNA%3D%3D&devicetype=Windows+10+x64&version=63030532&lang=zh_CN&exportkey=A155QDCEwTpkketUqpAKgGY%3D&pass_ticket=kS%2By0c8li7Rek8HpSLftJF1fgxQTiBBzrDZgJmY5sZserNzJ%2FrwNgVGciPuo%2FufJ&wx_header=0&fontgear=2)
- [详细全面代码](https://nlp.seas.harvard.edu/2018/04/03/attention.html)

### 2. 核心模块

#### 2.1 Transformer结构

##### 2.1.1 Transformer

- Encoder
  - EncoderLayer
    - Token Embeddings；Positional Encodings
    - Multi-Head Attention
    - Dropout
    - Residual Connection
    - LayerNorm
    - FeedForward Network
    - Dropout
    - Residual Connection
    - LayerNorm
- Decoder
  - DecoderLayer
    - Token Embeddings；Positional Encodings
    - Masked Multi-Head Attention：tgt_mask
    - Dropout
    - Residual Connection
    - LayerNorm
    - Multi-Head Attention：src_mask；decoder_hidden_states：query，encoder_hidden_states：key、value
    - Dropout
    - Residual Connection
    - LayerNorm
    - FeedForward Network
    - Dropout
    - Residual Connection
    - LayerNorm

#### 2.2 位置编码

目前常用的位置编码方式（The Positional Encodings or Positional Embeddings）包含如下几种：

1. 绝对位置编码（ absolute position）
   - 三角式：Transformer采用
   - 可学习式：可学习嵌入位置编码，（Learned Positional Embeddings）BERT采用，
2. 相对位置编码（ relative position）
   - 建模序列中token两两之间的相对距离，Transformer-XL 采用。
3. 旋转位置编码（RoPE）
   - ChatGLM、LLaMA、PaLM采用，绝对编码实现相对编码

Transformer中采用的位置编码方案如下：
$$
PE(pos,2i)=sin(pos/10000^{2i/d_{model}}) \\
PE(pos,2i+1)= cos(pos/10000^{2i/d_{model}})
$$
可根据上述公式生成每个位置的表示，对应位置$[0,1,2,3,…,511]$，其表示形式如：$[sin,cos,sin,cos,…,cos]$

计算变形：
$$
\large pos/10000^{2i/d_{model}} = pos · e^{2i·log(10000.0)/d_{model}}
$$
原论文中有该编码方式与可学习嵌入编码方式的效果进行对比，实验结果是两者效果几乎一样，但因为该编码方式可以允许模型将位置编码推广到比训练时遇到的序列长度更长的序列，因而采用了该编码方式。

实现方式一：

```python
def get_position_encodings1(max_position_embeddings, hidden_size):
    position_encodings = torch.zeros(max_position_embeddings, hidden_size)
    for pos in range(max_position_embeddings):
        for i in range(0, int(hidden_size//2)):
            item = pos / (10000 ** (2*i / hidden_size))
            new_item = pos * math.exp((float(2*i) * (-math.log(10000.0) / hidden_size)))
            position_encodings[pos, 2*i] = math.sin(item)
            position_encodings[pos, 2*i + 1] = math.cos(item)
    return position_encodings

max_position_embeddings, hidden_size = 6, 4
position_encodings = get_position_encodings(max_position_embeddings, hidden_size)
print(position_encodings)
```

实现方式二：

```python
def get_position_encodings(max_position_embeddings, hidden_size):

    position_encodings = torch.zeros(max_position_embeddings, hidden_size)
    for pos in range(max_position_embeddings):
        for i in range(0, hidden_size, 2):
            item = pos / (10000 ** (i / hidden_size))
            new_item = pos * math.exp((float(i) * (-math.log(10000.0) / hidden_size)))
            position_encodings[pos, i] = math.sin(item)
            position_encodings[pos, i + 1] = math.cos(item)
    return position_encodings

max_position_embeddings, hidden_size = 6, 4
position_encodings = get_position_encodings(max_position_embeddings, hidden_size)
print(position_encodings)
```

实现方式三：

```python
def get_position_encodings(max_position_embeddings, hidden_size):
    position_encodings = torch.zeros(max_position_embeddings, hidden_size)
    position = torch.arange(0, max_position_embeddings, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
    position_encodings[:, 0::2] = torch.sin(position * div_term)
    position_encodings[:, 1::2] = torch.cos(position * div_term)
    return position_encodings

max_position_embeddings, hidden_size = 6, 4
position_encodings = get_position_encodings(max_position_embeddings, hidden_size)
print(position_encodings)
```

#### 2.3 MASK 使用

> Masking plays an important role in the transformer. It serves two purposes:
>
> - In the encoder and decoder: To zero attention outputs wherever **there is just padding in the input sentences**.
> - In the decoder: **To prevent the decoder from conditioning to future or subsequent tokens** when predicting the next token.

##### 2.3.1 针对输入 padding 进行 MASK

```python
input_ids = torch.as_tensor([
    [9, 15, 6, 3, 9, 0, 0, 0],
    [18, 16, 32, 68, 95, 39, 26, 0]
])

pad_token_id = 0

# creates mask with 0s wherever there is padding in the input
input_mask = torch.as_tensor(input_ids != pad_token_id,dtype=torch.long).unsqueeze(1)
input_pad_mask = input_ids != pad_token_id
print(input_mask)

```

在多头注意力机制内部会对 mask 进行如下后处理：

```python
if mask is not None:
    if mask.dim() == 2:
        mask = mask.unsqueeze(1).unsqueeze(2)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
	attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
```

实际上 `masked_fill` 执行的就是如下操作：

```python
mask = mask.to(attention_scores.dtype) # fp16 compatibility
attention_scores = attention_scores * mask - (1 - mask) * 1e9
```



##### 2.3.2 针对subsequent tokens 进行 MASK

该 MASK操作仅在Decoder中 **Masked Multi-Head Attention** 模块使用，保证当前token无法看到序列中在其之后的token。

**使用numpy实现**

```python
def get_tgt_mask(tgt_input_ids, tgt_pad_token_id=0):
    tgt_seq_len = tgt_input_ids.size(1)
    # 首先对输入中的padding部分进行MASK处理
    tgt_input_mask = torch.as_tensor(tgt_input_ids != tgt_pad_token_id, dtype=torch.long)
    
    # 生成上三角全为1的矩阵
    subsequent_mask = np.triu(np.ones((1, tgt_seq_len, tgt_seq_len)), k=1).astype("uint8")
    # 转换为下三角全为1的矩阵
    subsequent_mask = torch.from_numpy(subsequent_mask) == 0
    
    # 将两种mask结合
    tgt_mask = tgt_input_mask.unsqueeze(1) & subsequent_mask
    print(tgt_mask.shape) # (batch_size,tgt_seq_len, tgt_seq_len)
    return tgt_mask
```

**使用torch实现：推荐使用**

```python
def get_tgt_mask(tgt_input_ids, tgt_pad_token_id=0):
    tgt_seq_len = tgt_input_ids.size(1)
    tgt_input_mask = torch.as_tensor(tgt_input_ids != tgt_pad_token_id, dtype=torch.long)
	# 取矩阵下三角以及对角线元素，其余元素变为0
    subsequent_mask = torch.tril(torch.ones(1,tgt_seq_len, tgt_seq_len)).long()
    
    # 取矩阵上三角以及对角线元素，其余元素变为0
    # subsequent_mask = torch.triu(torch.ones(1, tgt_seq_len, tgt_seq_len), diagonal=1) == 0
    tgt_mask = tgt_input_mask.unsqueeze(1) & subsequent_mask
    print(tgt_mask.shape)
    return tgt_mask
```

- 工具函数

  - `torch.tril()`：取下三角以及对角线元素，diagonal=0，包括对角线，diagonal=1则不包括

    ```markdown
       tril(input, diagonal=0, *, out=None) -> Tensor
        
        Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices
        :attr:`input`, the other elements of the result tensor :attr:`out` are set to 0.
        
        The lower triangular part of the matrix is defined as the elements on and
        below the diagonal.
        
        The argument :attr:`diagonal` controls which diagonal to consider. If
        :attr:`diagonal` = 0, all elements on and below the main diagonal are
        retained.
        
    ```

  - `torch.triu()`：取上三角以及对角线元素，diagonal=0，包括对角线，diagonal=1则不包括

    ```markdown
     triu(input, diagonal=0, *, out=None) -> Tensor
        
        Returns the upper triangular part of a matrix (2-D tensor) or batch of matrices
        :attr:`input`, the other elements of the result tensor :attr:`out` are set to 0.
        
        The upper triangular part of the matrix is defined as the elements on and
        above the diagonal.
        
        The argument :attr:`diagonal` controls which diagonal to consider. If
        :attr:`diagonal` = 0, all elements on and above the main diagonal are
        retained. 
    ```

    

- 参考PyTorch官方版本

  版本1：

  ```python
  def generate_square_subsequent_mask(self, sz):
      mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
      mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
      return mask
  ```

  版本2：

  ```python
  def generate_square_subsequent_mask(sz: int) -> Tensor:
      """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
      return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
  ```

**底层实现**

```python
import random
import torch

def get_extended_attention_mask(attention_mask):
    """可以直接由attention_mask,生成结合subsequent mask后的mask"""
    batch_size, seq_length = attention_mask.size()
    seq_ids = torch.arange(seq_length)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    # in case past_key_values are used we need to add a prefix ones mask to the causal mask
    # causal and attention masks must have same type with pytorch version < 1.3
    causal_mask = causal_mask.to(attention_mask.dtype)
    
    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    
    # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


tgt = torch.as_tensor(
    [
        [random.choice(list(range(72))) for _ in range(150)] + [0] * 22,
        [random.choice(list(range(72))) for _ in range(156)] + [0] * 16,
    ]
)

attention_mask = torch.as_tensor(tgt !=0,dtype=torch.long)
print(get_extended_attention_mask(attention_mask).size())
```

- BERT源码参考

```python
from transformers.modeling_utils import ModuleUtilsMixin

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask
```

#### 2.4 注意力机制

##### 2.4.1  Scaled Dot-Product Attention

- 计算公式
  $$
  Attention(Q,K,V)=softmax(\frac{QK^\mathrm T}{\sqrt{d_k}})V
  $$
  
  关于为何要除以 $\sqrt{d_k}$ ，原论文中有如下解释：
  
  > We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients . To counteract this effect, we scale the dot products by $1/\sqrt{d_k}$.
  >
  > 用于避免点积结果较大，导致梯度变得太小。

结论：

- key和value的序列长度需要保持相同，因为需要进行加权求和，权重的个数等于value的序列长度。

- query 和 key需要满足向量维度相同，因为需要进行向量内积计算，而query和key的序列长度不必保持相同。

  如在Transformer的Decoder中就存在解码器隐藏状态输出decoder_hidden_states作为query，而编码器最后一层隐藏状态输出encoder_hidden_states同时作为key和value的情况，此时query和key的序列长度就不相等。

  > The decoder multi-head attention layer **uses the decoder representation as the query and the encoder representation as the key and value.**

假定 ：

- decoder_hidden_states的张量形状为：（batch_size, tgt_seq_len, hidden_size），对应 $Q$

- encoder_hidden_states的张量形状为：（batch_size, src_seq_len, hidden_size），对应 $K$

- $Q$ 和 $K$ 矩阵相乘得到个attention_scores 的张量形状为：（batch_size, tgt_seq_len, src_seq_len）

- attention_scores矩阵的行表示query对应的token序列，列表示key对应的token序列，且每一行求和的结果都为1。

  ![quey_and_key](D:\Typora\Notes\NLP\经典模型\quey_and_key.png)

- 对于attention_scores矩阵，若key序列中k3是填充的token，则k3那一列对应的值都为0，这是MASK作用的结果。

- 若attention_scores矩阵为Masked Multi-Head Attention计算得到的矩阵，则当前token都无法看到在其之后的token，也就是q1只能看到k1，q2只能看到k1、k2，q3只能看到k1、k2、k3，……，从而导致attention_scores矩阵的上三角元素值都为0，这也是MASK作用的结果。如下图所示：

  ![掩码](D:\Typora\Notes\NLP\经典模型\掩码.png)

  

### 3 结构优化

#### 3.1 对LayerNorm层进行优化

- 论文：

  - Attention Is All You Need  对应Post-LN
  - On Layer Normalization in the Transformer Architecture   对应Pre-LN
  - CogView: Mastering Text-to-Image Generation via Transformers  对应 Sandwich-LN
  - DeepNet: Scaling Transformers to 1,000 Layers  对应DeepNorm

- 优化细节

  ![LayerNorm变形](D:\Typora\Notes\NLP\经典模型\LayerNorm变形.png)

  

- 原始的Transformer、BERT论文采用 Post-LN结构，但Post-LN结构容易发散，近年来模型普遍采用Pre-LN结构（出自ChatGLM视频报告）

- 数百亿/多模态混合精度训练（FP16）中，Pre-LN也不稳定

  - Pre-LN的变体Sandwich-LN结构可以缓解这一现象

- DeepNet：调整残差，修改初始化，从而稳定千层Post-LN
  $$
  DeepNorm(x) = LayerNorm(α·x+g(x)),  α>1
  $$

- GLM130B规模实验：DeepNorm比Sandwich-LN更稳定，但Post-LN比Pre-LN对超参数更敏感，需要仔细调试

![image-20230607170315801](C:\Users\千江映月\AppData\Roaming\Typora\typora-user-images\image-20230607170315801.png)























词汇表扩充



