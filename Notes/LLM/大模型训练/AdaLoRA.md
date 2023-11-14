## AdaLoRA

[讲解](https://readpaper.feishu.cn/docx/B14mdPP23olvNfxnJLncyNvenVd)

LoRA存在的问题：预先指每个增加矩阵的内在秩，忽略了在微调预训练模型时，权重矩阵的重要性在不同模块和层之间存在显著差异。针对LoRA存在的问题，其提出动态更新增加矩阵的秩，同时找到更加重要的矩阵，分配更多的参数，提升模型效果，裁剪不重要的矩阵，降低参数计算量，降低模型表现差的风险。

LoRA方法通常在所有微调的权重矩阵上均匀地分配增量更新预算，也就是都使用相同的秩，却忽略了不同权重参数的不同重要性。

AdaLoRA adaptively allocates the parameter budget（rank） among weight matrices according to their importance score.

SVD：
$$
M =  U·Σ·V^T = \sum_{i=1}^r \sigma_i u_iv_i
$$
