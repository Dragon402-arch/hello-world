## Bloom Model 

激活函数：GeLU

模型结构：（参数取自Tigetbot模型）

```shell
BloomForCausalLM(
  (transformer): BloomModel(
    (word_embeddings): Embedding(250682, 4096)
    (word_embeddings_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
    (h): ModuleList(
      (0-29)x30: BloomBlock(
        (input_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (self_attention): BloomAttention(
          (query_key_value): Linear(in_features=4096, out_features=12288, bias=True)
          (dense): Linear(in_features=4096, out_features=4096, bias=True)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (post_attention_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (mlp): BloomMLP(
          (dense_h_to_4h): Linear(in_features=4096, out_features=16384, bias=True)
          (gelu_impl): BloomGelu()
          (dense_4h_to_h): Linear(in_features=16384, out_features=4096, bias=True)
        )
      )
    (ln_f): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=4096, out_features=250682, bias=False)
)

```



### 1 Mask 操作

#### 1.1 pad mask 

```python
import torch

def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    # 取反操作，也就是将False变为True
    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    """
    [False, False, False]
    [False, False, False]
    """
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)
```



#### 1.2 casual mask

```python
import torch


def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.empty(
        (target_length, target_length + past_key_values_length),
        dtype=torch.bool,
        device=device,
    )
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    # 写法优秀
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]
    
    """
    [False,True,True]
    [False False,True]
    [False,False,False]
    """

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(
        batch_size, 1, target_length, target_length + past_key_values_length
    )
    return expanded_mask
```

#### 1.3 合并

```python
def prepare_attn_mask(
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        past_key_values_length: int,
) -> torch.BoolTensor:
    # create causal mask
    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
    combined_attention_mask = None
    device = attention_mask.device
    _, src_length = input_shape

    if src_length > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape, device=device, past_key_values_length=past_key_values_length
        )

    # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
    expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
    combined_attention_mask = (
        expanded_attn_mask
        if combined_attention_mask is None
        else expanded_attn_mask | combined_attention_mask 
        # A|B 表示相同位置有一个True就是True，都是False才是False
    )

    return combined_attention_mask

device = torch.device("cuda")
attention_mask = torch.ones((2, 8), device=device)
attention_mask[:, 6:] = 0
print(attention_mask)
input_shape = attention_mask.shape
past_key_values_length = 0
out = prepare_attn_mask(attention_mask, input_shape, past_key_values_length)
```

### 2 位置编码

#### 2.1 Alibi

slopes 计算公式：
$$
slopes = \Large  2^{\left ( -\frac {\Huge 8i} {\Huge n}\right)}
$$

```python
def build_alibi_tensor(
    attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
) -> torch.Tensor:
    """
    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))

    """计算公式：2^(-8/n)"""
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        device=attention_mask.device,
        dtype=torch.float32,
    )
    #
    powers = torch.arange(
        1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32
    )
	# base 作为计算每个head超参数的基础，然后根绝head为1/2/3/4等，取对应的次方即可。
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            device=attention_mask.device,
            dtype=torch.float32,
        )
        # 一般都是取 num_heads - closest_power_of_2
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(
            1,
            1 + 2 * num_remaining_heads,
            2,
            device=attention_mask.device,
            dtype=torch.int32,
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    print(arange_tensor)
    alibi = slopes[..., None] * arange_tensor
    print(alibi.shape)
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)
```

测试

```python
attention_mask = torch.ones((2, 16))
attention_mask[:, 14:] = 0
dtype = torch.float
num_heads = 3

build_alibi_tensor(attention_mask, num_heads, dtype)
```

#### 2.2 ALiBi使用

```python
class BloomAttention(nn.Module):
    def __init__(self,):
         self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
         self.beta = 1.0
        
    def forward(
        self,
    ): # [batch_size * num_heads, q_length, kv_length]
        # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` 
        # as the latter isn't supported by TorchScript v1.11
        matmul_result = alibi.baddbmm(
            batch1=query_layer,
            batch2=key_layer,
            beta=self.beta, # beta值需要乘上alibi
            alpha=self.inv_norm_factor,
        )
        

class BloomModel(BloomPreTrainedModel):
    
    def build_alibi_tensor(
        self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype
    ) -> torch.Tensor:
        return build_alibi_tensor(attention_mask, num_heads, dtype)

    def forward(
        self,
    ):
        alibi = self.build_alibi_tensor(
            attention_mask, self.num_heads, dtype=hidden_states.dtype
        )
```

注解：`torch.Tensor.baddbmm()`

```python
input.baddbmm(batch1, batch2, *, beta=1, alpha=1, out=None)
	Performs a batch matrix-matrix product of matrices in :attr:`batch1`and :attr:`batch2`.
    :attr:`input` is added to the final result.
    :attr:`batch1` and :attr:`batch2` must be 3-D tensors each containing the same number of matrices.
                    
```

$$
      \text{out}_i = \beta\ \text{input}_i + \alpha\ (\text{batch1}_i \mathbin{@} \text{batch2}_i)
$$

其中 `@` 表示矩阵乘法

### 3 残差连接位置

相关代码：

残差连接施加的对象是经过LayerNorm之前的，还是之后的，一般情况下都是使用经过LayerNorm之前的。

```python
class BloomBlock(nn.Module):
    def forward(self,):
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else: # GPT-2 使用的是这个分支
            residual = hidden_states
            
        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else: # GPT-2 使用的是这个分支
            residual = attention_output

```



### 4 MHSA 和 MLP 输出转换

矩阵直接求和（如：(2,1024,4096) x (4096,1024)），矩阵分块求和再相加（(2,1024,2048:4096) x (2048:4096,1024)+(2,1024,2048:4096) x (2048:4096,1024) ），两者的结果由于数据表示的问题会存在微小的差异。图示说明：

![tensor-parallel](D:\Typora\Notes\LLM\经典大模型\tensor-parallel.png)

测试代码：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 10
hidden_dim = 512
output_dim = 512
sliced_dim = hidden_dim//2

dummy_mlp = nn.Linear(hidden_dim, output_dim, bias=False)
dummy_input = torch.randn(batch_size, hidden_dim)

sliced_input_1, sliced_input_2 = torch.split(dummy_input, sliced_dim, dim=-1)

sliced_output_1 = F.linear(sliced_input_1, dummy_mlp.weight[:, :sliced_dim])
sliced_output_2 = F.linear(sliced_input_2, dummy_mlp.weight[:, sliced_dim:])

final_output = dummy_mlp(dummy_input)

torch.testing.assert_close(final_output, sliced_output_1 + sliced_output_2, rtol=0.0, atol=0.0)

```

大模型的知识库倒是接触了两个开源项目，一个dify,一个fastgpt。可以自己塞相关的专业知识给他，不过有问题，暂时没看到俩家对接其他模型。fastgpt看到有人对接6B的模型了。

#### 4.1 MHSA部分转换

相关代码

```python
# aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
if self.pretraining_tp > 1 and self.slow_but_exact:
    slices = self.hidden_size / self.pretraining_tp
    output_tensor = torch.zeros_like(context_layer)
    for i in range(self.pretraining_tp):
        output_tensor = output_tensor + F.linear(
            context_layer[:, :, int(i * slices): int((i + 1) * slices)],
            self.dense.weight[:, int(i * slices): int((i + 1) * slices)],
        )
else:
	output_tensor = self.dense(context_layer)
```

#### 4.2 MLP部分转换

```python
# hidden_states是dense_h_to_4h的输出。

if self.pretraining_tp > 1 and self.slow_but_exact:
    intermediate_output = torch.zeros_like(residual)
    slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
    for i in range(self.pretraining_tp):
        intermediate_output = intermediate_output + F.linear(
            # hidden_states shape:(2,1024,4096)
            hidden_states[:, :, int(i * slices): int((i + 1) * slices)],
            self.dense_4h_to_h.weight[:, int(i * slices): int((i + 1) * slices)],
            # self.dense_4h_to_h.weight shape:(1024,4096)
            # (2,1024,2048:4096) x (2048:4096,1024)
        )
else:
     intermediate_output = self.dense_4h_to_h(hidden_states)
        
```



