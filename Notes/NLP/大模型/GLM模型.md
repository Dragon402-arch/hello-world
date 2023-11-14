### ChatGLM

#### 模型结构

```shell
ChatGLMForConditionalGeneration(
  (transformer): ChatGLMModel(
    (word_embeddings): Embedding(150528, 4096)
    (layers): ModuleList(
      (0-27): 28 x GLMBlock(
        (input_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (attention): SelfAttention(
          (rotary_emb): RotaryEmbedding()
          (query_key_value): Linear(in_features=4096, out_features=12288, bias=True)
          (dense): Linear(in_features=4096, out_features=4096, bias=True)
        )
        (post_attention_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (mlp): GLU(
          (dense_h_to_4h): Linear(in_features=4096, out_features=16384, bias=True)
          (dense_4h_to_h): Linear(in_features=16384, out_features=4096, bias=True)
        )
      )
    (final_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=4096, out_features=150528, bias=False)
)
```



### ChatGLM2

#### 模型结构

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



- 

- 

- 

- 

- 

- 

- 

- 

- In practice, GLM-130B uses two different mask tokens (`[MASK]` and `[gMASK]`) for short and long text generation, respectively.

  - [MASK] : 512
  - [gMASK] : 2048
  - During the 60-day access to the cluster, we manage to train GLM-130B for 400 billion tokens (roughly 200 billion each for Chinese and English) with a fifixed sequence length of 2,048 per sample.For the [gMASK] training objective, we use a context window of 2,048 tokens. For the [MASK] and multi-task objectives, we use a context window of 512 and concatenate four samples together to cater the 2,048-sequence-length.

- Rotary positional encoding (RoPE)

- DeepNorm——layer normalization

- 模型尺寸

  - max_seq_len:2048
  - hidden_size:12288
  - num_layers:70

- eop:end of  paragraphs

- sop:start of  paragraphs

- 模型

  - sparse model：稀疏模型
    - mixture-of-experts (MoE)
  - dense model：稠密模型

- tokenizer

  - common tokens：20000 - 20099，包括标点符号、数字、空格
  - English tokens：20100 - 83822
  - Chinese tokens：83823 - 145653
  - other

- special tokens

  -  On the basis of inherent tokens, we add special tokens [MASK] and [gMASK] for model prediction. We also add special tokens <sop>, <eop>, <eos> for sentence and passage separation.

    ```json
    		{'bos_token': '<sop>', 
    		'eos_token': '</s>', 
    		'unk_token': '<unk>', 
    		'pad_token': '<pad>', 
    		'mask_token': '[MASK]'}
    ```

    [token, ..., token, gmask, bos, token, ... token, eop]

警情文档查询