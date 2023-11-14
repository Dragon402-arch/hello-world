### FlashAttention

![FlashAttention](D:\Typora\Notes\LLM\大模型组件\FlashAttention.webp)

Flash要点：

- **Fast** — excerpt from the paper: “We train BERT-large (seq. length 512) 15% faster than the training speed record in MLPerf 1.1, GPT2 (seq. length 1K) 3x faster than baseline implementations from HuggingFace and Megatron-LM, and long-range arena (seq. length 1K-4K) 2.4x faster than baselines.”
- **Memory-efficient** — compared to vanilla attention, which is quadratic in sequence length, *O(N²)*, this method is sub-quadratic/linear in N (*O(N)*).
- **Exact** — meaning it’s not an approximation of the attention mechanism (like e.g. sparse, or low-rank matrix approximation methods) — its outputs are the same as in the “vanilla” attention mechanism.
- **IO aware** — compared to vanilla attention, flash attention is **sentient**.

[理论基础](https://readpaper.feishu.cn/docx/AC7JdtLrhoKpgxxSRM8cfUounsh)：

在传统算法中，一种方式是将Mask和SoftMax部分融合，以减少访存次数。然而，FlashAttention则更加激进，它将从输入$Q、K、V$到输出$O$的整个过程进行融合，以避免attention score及其经过softmax之后的矩阵的存储开销，实现端到端的延迟缩减。然而，由于输入的长度$$N$$通常很长，无法完全将完整的$Q、K、V、O$及中间计算结果存储在SRAM中。因此，需要依赖HBM进行访存操作，与原始计算延迟相比没有太大差异，甚至会变慢（没具体测）。

为了让计算过程的结果完全在SRAM中，摆脱对HBM的依赖，可以采用**分片操作**，每次进行部分计算（分块计算），确保这些计算结果能在SRAM内进行交互，待得到对应的结果后再进行输出。

[算法标准流程图](https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad)：

![FlashAttention算法流程](D:\Typora\Notes\LLM\大模型组件\FlashAttention算法流程.webp)

算法流程中除去数值稳定性处理后核心部分就是：
$$
S_{ij} = Q_iK^T_j \in R^{\large B_r * B_c} \\
O_i = Q_iK^T_j V_j = S_{ij} V_j \in R^{\large B_r * d}
$$
如 Q：64 x 128，K，V：64 x 128，输出结果 O：64 x 128，$B_r=B_c=8$，而每次计算得到的$S_{ij}$：8 x 8， $O_i$ ：8 x 128，每次计算得到的 $O_i$ 要在上次计算结果的基础上进行更新，直至内循环结束。

- 目标输出 O：64 x 128
- 每次内循环结束 $O_i$ ：8 x 128
- 外循环经过8个内循环后，就可以将目标输出 O 填充完毕。

FlashAttention[计算简化过程](https://ahmdtaha.medium.com/flashattention-fast-and-memory-efficient-exact-attention-with-io-awareness-2a0aec52ed3d)（去除了数值稳定性部分）：

![FlashAttention](D:\Typora\Notes\LLM\大模型组件\FlashAttention.png)

其中$D_b$ 表示当前block分母中的一项， $D$ 表示softmax函数表达式中的分母，也就是一行元素的分母，$O$ 表示注意力机制输出结果。

代码实现：

```python
import math

import torch

Q = [1, 5, 9]
K = [1, 2, 3]
V = [2, 4, 8]
D = [0] * 3
O = [0] * 3


def flash_attention(Q, K, V):
    for j in range(len(K)):
        k, v = K[j], V[j]
        for i in range(len(Q)):
            q = Q[i]
            D_b = math.exp(q * k)  # current block denominator 分母
            N_b = v * D_b  # current block numerator 分子
            O[i] = (D[i] * O[i] + N_b) / (D[i] + D_b)
            D[i] += D_b
    return O


def source_attention(Q, K, V):
    Q = torch.as_tensor(Q, dtype=torch.float)[:, None]
    K = torch.as_tensor(K, dtype=torch.float)[:, None]
    V = torch.as_tensor(V, dtype=torch.float)[:, None]
    attn_score = torch.matmul(Q, K.transpose(0, 1))
    attn_score = torch.softmax(attn_score, dim=-1)
    output = torch.matmul(attn_score, V).squeeze().tolist()
    return output


print(source_attention(Q, K, V), flash_attention(Q, K, V))

```

