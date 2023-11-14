[教程](https://www.pinecone.io/learn/batch-layer-normalization/)



### 1. Batch Normalization

 **For each feature, batch normalization computes the mean and variance of that feature in the mini-batch. It then subtracts the mean and divides the feature by its mini-batch standard deviation.**

计算过程图示：

![batch-normalization-example](D:\Typora\Notes\NLP\batch-normalization-example.png)

![BatchNorm](D:\Typora\Notes\LLM\大模型组件\BatchNorm.png)

计算公式：
$$
\hat {x} = \frac{\vec x - \mathrm{E}[\vec x]}{ \sqrt{\mathrm{Var}[\vec x] + \epsilon}} \odot \vec\gamma + \vec \beta
$$

其中 $\odot$  表示哈达玛积（Hadamard product），是一种矩阵运算，运算时将矩阵对应位置的元素相乘。除  $\epsilon$ 外公式中其余的都是向量表示。

### 2. Layer Normalization

Layer normalization **normalizes input across the features** instead of **normalizing input features across the batch** dimension in batch normalization.

- 计算图示

  ![layer-normalization](D:\Typora\Notes\NLP\layer-normalization.png)

  

- 计算公式：
  $$
  \hat {x} = \frac{\vec x - \mathrm{E}[\vec x]}{ \sqrt{\mathrm{Var}[\vec x] + \epsilon}} \odot \vec\gamma + \vec \beta
  $$
  
  其中 $\odot$  表示哈达玛积（Hadamard product），是一种矩阵运算，运算时将矩阵对应位置的元素相乘。除  $\epsilon$ 外公式中其余的都是向量表示。

### 3. DeepNorm

- 计算公式：
  $$
  \rm DeepNorm(x) = \rm LayerNorm(\alpha·x+g(x)),  \alpha>1
  $$
  其中 $\alpha$ 是一个常数，其中 $g(·)$ 表示Transformer Sub-layer（Attention or feed-forward network）。

- 应用：GLM-130B

### 4. RMSNorm

**Root mean square layer normalization** (RMSNorm), which regularizes the summed inputs to a neuron in one layer with the root mean square (RMS) statistic alone.
$$
\hat x = \frac{ \vec x}{ \sqrt{\mathrm{Var}[ \vec x] + \epsilon}} \odot \vec\gamma 
$$

$$
\mathrm{Var}[ \vec x] = \sqrt{\frac{1}{n}\sum_{i=1}^n{ \vec x_i^2}}
$$

When the mean of summed inputs is zero, RMSNorm is exactly equal to LayerNorm.







