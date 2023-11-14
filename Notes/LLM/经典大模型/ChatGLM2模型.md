## ChatGLM2 Model

### 1. 模型结构

```shell
  (transformer): ChatGLMModel(
    (embedding): Embedding(
      (word_embeddings): Embedding(65024, 4096)
    )
    (rotary_pos_emb): RotaryEmbedding()
    (encoder): GLMTransformer(
      (layers): ModuleList(
        (0-27): 28 x GLMBlock(
          (input_layernorm): RMSNorm()
          (self_attention): SelfAttention(
            (query_key_value): Linear(in_features=4096, out_features=4608, bias=True)
            (core_attention): CoreAttention(
              (attention_dropout): Dropout(p=0.0, inplace=False)
            )
            (dense): Linear(in_features=4096, out_features=4096, bias=False)
          )
          (post_attention_layernorm): RMSNorm()
          (mlp): MLP(
            (dense_h_to_4h): Linear(in_features=4096, out_features=27392, bias=False)
            (dense_4h_to_h): Linear(in_features=13696, out_features=4096, bias=False)
          )
        )
      )
      (final_layernorm): RMSNorm()
    )
    (output_layer): Linear(in_features=4096, out_features=65024, bias=False)
  )
)
```

### 2. 核心模块

#### 2.1 Attention 

经典 Attention 模块及其变形示意图：

![GQA](D:\Typora\Notes\LLM\经典大模型\GQA.png)

MQA：only uses a single key-value head

GQA：uses an intermediate (more than one, less than number of query heads) number of key-value heads.

Grouped-query attention divides query heads into *G*  groups, each of which shares a single key head and value head.

Multi-Query Attention is a powerful and efficient alternative to the traditional Multi-Head Attention mechanism. By **sharing Key and Value vectors across multiple heads**, it offers reduced memory footprint and computational complexity, making it a compelling choice for performance-critical applications

group_num = 2，head_dim = 64，num_heads =12

相关代码

```python
class SelfAttention(torch.nn.Module):
    """Parallel self-attention layer abstract class.
    Self-attention layer takes input with size [b,s,h] and returns output of the same size.
    """

    def __init__(self, config: ChatGLMConfig, layer_idx, device=None):
        super(SelfAttention, self).__init__()
        self.layer_idx = max(1, layer_idx)

        self.projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        self.hidden_size_per_attention_head = (
                self.projection_size // config.num_attention_heads
        )
        # partition 也即是指 query / key / value
        self.num_attention_heads_per_partition = config.num_attention_heads

        self.multi_query_attention = config.multi_query_attention
        # 正常情况
        self.qkv_hidden_size = 3 * self.projection_size
        if self.multi_query_attention:
            # 也即是每个key和value使用了几个 head
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            """
            GQA情况：
            	query:
            		kv_channels(self.hidden_size_per_attention_head)
            		self.num_attention_heads_per_partition
            	key / value:
            		kv_channels(self.hidden_size_per_attention_head)
            		config.multi_query_group_num
            """
            self.qkv_hidden_size = (
                    self.projection_size
                    + 2 * self.hidden_size_per_attention_head * config.multi_query_group_num
            )
	 def forward(
            self,
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            kv_cache=None,
            use_cache=True,
    ):
        mixed_x_layer = self.query_key_value(hidden_states)
        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition
                    * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition
                    * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition
                    * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1]
                + (
                    self.num_attention_heads_per_partition, # 2 
                    self.hidden_size_per_attention_head, # 128
                )
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            """
            torch.Size([1, 1, 32, 128]) torch.Size([1, 1, 2, 128]) torch.Size([1, 1, 2, 128])
            """
```

#### 2.2 RoPE

相关代码：[参考](https://nn.labml.ai/transformers/rope/index.html)

```python
@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x:  (batch_size,seq_len,num_heads,head_dim)
        rope_cache:(batch_size,seq_len,head_dim//4,2)
    Return:
    """
    batch_size, seq_len, num_heads, head_dim = x.shape
    rope_dim = rope_cache.shape[-2] * 2

    rope_cache = rope_cache[:, :seq_len]
    rope_cache = rope_cache.view(-1, seq_len, 1, rope_dim // 2, 2)
    rope_cos, rope_sin = rope_cache[..., 0], rope_cache[..., 1]

    # Split the features, we can choose to apply rotary embeddings only to a partial set of features.
    x_rope, x_pass = x[..., :rope_dim], x[..., rope_dim:]
    x_rope_down, x_rope_up = x_rope[..., rope_dim // 2:], x_rope[..., : rope_dim // 2]

    x_rope = torch.stack(
        [
            x_rope_down * rope_cos - x_rope_up * rope_sin,
            x_rope_up * rope_cos + x_rope_down * rope_sin,
        ],
        -1,
    )
    x_rope = x_rope.flatten(3)
    return torch.cat((x_rope, x_pass), dim=-1)


class ChatGLMModel(ChatGLMPreTrainedModel):
    self.rotary_pos_emb = RotaryEmbedding(rotary_dim // 2, original_impl=config.original_rope, device=device,
                                          dtype=config.torch_dtype)
    def forward(self,):
        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]

```

调用：

```python
class SelfAttention(torch.nn.Module):  
    def forward(self,)
    # apply relative positional encoding (rotary embedding)
    if rotary_pos_emb is not None:
        query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)
```





**The FlashAttention algorithm** has been instrumental in this by speeding up attention and reducing memory consumption for even longer sequences for the attention layer. Moreover, the model has been trained with a **context length of 8K** during the dialogue alignment to offer users more conversational depth. ChatGLM2-6B also uses the **Multi-Query Attention technique**, thereby successfully achieving lower GPU memory usage of the **KV Cache** and increased inference speed, approximately 42%, compared to the first generation.

- FlashAttention：xformers
- VLLM’s PagedAttention

#### 2.3 MASK

推理时

```python
if attention_mask is None and query_seq_len == key_seq_len:
    # 推理时第一个token生成时使用，第二个token生成时query_seq_len == key_seq_len条件不满足
	ttention_mask = torch.ones(
            batch_size,
            1,
            query_seq_len,
            key_seq_len,
            device=attention_scores.device,
            dtype=torch.bool,
        )
        attention_mask.tril_()
        attention_mask = ~attention_mask
if attention_mask is not None:
    attention_scores = attention_scores.masked_fill(
        attention_mask, float("-inf")
    )
    attention_probs = F.softmax(attention_scores, dim=-1)
```







