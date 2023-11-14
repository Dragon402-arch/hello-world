### 1. Transformer mask

#### 1.1 pad mask

```python
def make_src_mask(self, src_input_ids):
    src_mask = (src_input_ids != self.src_pad_token_id).to(
        device=src_input_ids.device
    )
    return src_mask
```

#### 1.2 subsequent mask

```python
def make_tgt_mask(self, tgt_input_ids):
    tgt_seq_len = tgt_input_ids.size(1)

    tgt_pad_mask = (tgt_input_ids != self.tgt_pad_token_id).to(
        device=tgt_input_ids.device
    )

    # 取矩阵下三角以及对角线元素，其余元素变为0
    subsequent_mask = torch.tril(
        torch.ones(1, tgt_seq_len, tgt_seq_len, device=tgt_input_ids.device)
    ).bool()

    tgt_mask = tgt_pad_mask.unsqueeze(1) & subsequent_mask
    # print(tgt_mask.shape)
    return tgt_mask
```



### 2. BERT mask

**if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]**

#### 2.1 pad mask

源代码块取自：

```python
# past_key_values_length
past_key_values_length = (
    past_key_values[0][0].shape[2] if past_key_values is not None else 0
)

if attention_mask is None:
    attention_mask = torch.ones(
        ((batch_size, seq_length + past_key_values_length)), device=device
    )
# We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
# ourselves in which case we just need to make it broadcastable to all heads.
extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
    attention_mask, input_shape
)
    
def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
    if dtype is None:
        dtype = self.dtype
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
        return extended_attention_mask

```

#### 2.2 subsequent mask

BERT中对casual mask的写法

```python
def create_extended_attention_mask_for_decoder(input_shape, attention_mask, device=None):

    device = attention_mask.device
    batch_size, seq_length = input_shape
    seq_ids = torch.arange(seq_length, device=device)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    causal_mask = causal_mask.to(attention_mask.dtype)

    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    return extended_attention_mask
```



### 3. GPT2 mask

**if the model is a decoder, apply a causal mask in addition to the padding mask**

- self attention: 需要pad mask，也需要subsequent mask
- cross attention: 只需要pad mask 

#### 3.1 pad mask

源代码块写法取自：`class GPT2Model(GPT2PreTrainedModel)`

**self attention部分的pad mask**

```python
# GPT2Attention mask.
if attention_mask is not None:
    attention_mask = attention_mask.view(batch_size, -1)
    # We create a 3D attention mask from a 2D tensor mask.
    # Sizes are [batch_size, 1, 1, to_seq_length]
    # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
    # this attention mask is more simple than the triangular masking of causal attention
    # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
    attention_mask = attention_mask[:, None, None, :]

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
```

**cross attention 部分的 pad mask**

```python
    
# If a 2D or 3D attention mask is provided for the cross-attention
# we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
if self.config.add_cross_attention and encoder_hidden_states is not None:
    encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
    encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
    if encoder_attention_mask is None:
        encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
        encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)      
            
            
def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
    """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
    if encoder_attention_mask.dim() == 3:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
    if encoder_attention_mask.dim() == 2:
        encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(self.dtype).min

    return encoder_extended_attention_mask

```



#### 3.2 subsequent mask

源代码块写法取自:`class GPT2Attention`

```python
#  subsequent mask 部分使用
self.register_buffer(
        "bias",
        torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
            1, 1, max_positions, max_positions
        ),
    )

# 先对attn_weights进行subsequent mask操作
if not self.is_cross_attention:
    # if only "normal" attention layer implements causal mask
    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    # 获取 the dtype's smallest value，是一个负数 
    mask_value = torch.finfo(attn_weights.dtype).min # mask_value是一个普通标量值，如-3.4028234663852886e+38
    # 将 -3.4028234663852886e+38 转换为 tensor(-3.4028e+38)
    mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    # the dtype's smallest value for masked positions.
    attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
    
    # 等价于 attn_weights = attn_weights * causal_mask + mask_value * (1-causal_mask.float())
    # 等价于 attn_weights.masked_fill(causal_mask == 0, mask_value)


# 再对其进行pad mask 操作    
if attention_mask is not None:
    # Apply the attention mask
    attn_weights = attn_weights + attention_mask
```

注解示例

```python
causal_mask = torch.as_tensor([[1, 1, 1, 1, 0], [1, 1, 0, 0, 0]]).bool()
attention_scores = torch.rand(2, 5)
mask_value = torch.tensor(-188888.0)

# 以下三种写法效果相同
print(torch.where(causal_mask, attention_scores, mask_value))

print(attention_scores.masked_fill(causal_mask == 0, mask_value))
print(attention_scores.masked_fill(causal_mask == 0, float("-inf")))


print(attention_scores * causal_mask + mask_value * (1-causal_mask.float()))

"""
输出：
tensor([[ 7.0097e-01,  6.7522e-01,  6.7028e-02,  3.2576e-01, -1.8889e+05],
        [ 5.1675e-01,  2.9594e-01, -1.8889e+05, -1.8889e+05, -1.8889e+05]])
"""
```



