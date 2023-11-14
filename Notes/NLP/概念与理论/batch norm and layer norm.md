[教程](https://www.pinecone.io/learn/batch-layer-normalization/)

### Data Normalization

- 数据规范化

- 含义：Normalization is the process of rescaling the data so that it has same scale.**Data normalization transforms multiscaled data to the same scale. After normalization, all variables have a similar influence on the model, improving the stability and performance of the learning algorithm.**

- 目的：对不同变量或属性值进行缩放，从而使得所有属性在进行数据分析时保持相同的重要程度。

  Such transformation or mapping the data to a smaller or common range will help all attributes to gain equal weight. 

- 数据规范化的原因

  1. 如果数据没有规范化，一个特征可能完全支配其他特征。归一化使每个数据点具有相同的尺度，因此每个特征都同等重要。

  2. 它避免了对测量单位选择的依赖。

  3. 数据挖掘算法的应用变得更加简单、有效和高效。

  4. 更具体的数据分析方法可以应用于规范化数据。

  5. It prevent attributes with initially large ranges (e.g., income) from outweighing attributes with initially smaller ranges (e.g., binary attributes).

     它可以防止初始范围较大的属性(例如，收入)超过初始范围较小的属性(例如，二进制属性)。
     
  6. **An unintended benefit of Normalization is that it helps network in Regularization(only slightly, not significantly).**

- 为什么需要进行数据规范化

  Normalization is generally required **when we are dealing with attributes on a different scale**, otherwise, it may lead to a dilution in effectiveness of an important equally important attribute(on lower scale) because of other attribute having values on larger scale.

  In simple words, **when multiple attributes are there but attributes have values on different scales, this may lead to poor data models while performing data mining operations.** **So they are normalized to bring all the attributes on the same scale.**

  For example, consider a data set containing two features, age(x1), and income(x2). Where age ranges from 0–100, while income ranges from 0–100,000 and higher. Income is about 1,000 times larger than age. **So, these two features are in very different ranges.** When we do further analysis, like multivariate linear regression, for example, the attributed income will intrinsically influence the result more due to its larger value. But this doesn’t necessarily mean it is more important as a predictor. **So we normalize the data to bring all the variables to the same range**.

  ![data_norm](D:\Typora\Notes\NLP\data_norm.png)

  Variables that are measured at different scales do not contribute equally to the analysis and might end up creating a bais.For example, A variable that ranges between 0 and 1000 will outweigh a variable that ranges between 0 and 1. Using these variables without standardization will give the variable with the larger range weight of 1000 in the analysis. Transforming the data to comparable scales can prevent this problem.

  For machine learning,  not every dataset does require normalization. **It is required only when features have different ranges.** We make sure that the different features take on similar ranges of values **so that gradient descents can converge more quickly.**

  - 将不同特征的取值缩放到相近的范围，使得每个特征的重要程度等同。
  - 特征值缩放后，可以稳定学习（训练）过程（缩放前，相比于取值较小的特征值（年龄），取值较大的特征值（收入），会对训练过程产生较大的影响，从而导致了训练过程的不稳定性），大大降低了训练模型所需的epochs，梯度下降过程可以更快收敛。

- 数据规范化方法

  - Min-Max feature scaling，(often called **normalization**) ，归一化，对应 **MinMaxScaler**，对离群点比较敏感。

    - 标准计算公式：取值【0,1】
      $$
      x_{new} =  \frac{x-x_{min}}{x_{max}-x_{min}}
      $$

    - 变形计算公式：取值【a,b】
      $$
      x_{new} =  \frac{(x-x_{min})(b-a)}{x_{max}-x_{min}} + a
      $$
      以上计算公式，均是针对一个属性或是变量进行计算的，如果一个样本具有多个属性，则分别使用该公式进行计算即可。

      Rescaling data  to have values between 0 and 1.This is usually called feature scaling.

    - 注意：归一化后的数据分布与原数据分布不一致，

  - Z-Score feature scaling，(often called **standardization**) ，标准化，对应 **StandardScaler**

    - 计算公式：
      $$
      z_i = \frac{x_i-\overline x}{s} = \frac{x_i -μ }{σ}
      $$
      
    - 注意：
      1. 标准化后的数据均值为0，方差为1，
      2. 标准化后的数据与原始数据分布保持一致，但是不一定的是正态分布，当且仅当原始数据分布为正态分布时标准化后的数据才是标准正态分布。
  
- **Normalization vs. Standardization**

  The terms normalization and standardization are sometimes used interchangeably, but they usually refer to different things. **Normalization usually means to scale a variable to have a values between 0 and 1, while standardization transforms data to have a mean of zero and a standard deviation of 1.** This standardization is called a z-score.

### Batch Normalization

 **For each feature, batch normalization computes the mean and variance of that feature in the mini-batch. It then subtracts the mean and divides the feature by its mini-batch standard deviation.**

- [讲解](https://datahacker.rs/017-pytorch-how-to-apply-batch-normalization-in-pytorch/)

- 含义： It is a technique for training deep neural networks that standardizes the inputs to a layer for each mini-batch。

  针对一个特定的神经网络层的输入数据，标准化每个mini-batch数据的属性值。

- 为什么需要进行 batch norm

  当我们对数据集进行归一化并开始训练过程时，我们模型中的权重会在每个epcoh上更新。那么，如果在训练过程中，其中一个权重最终比另一个权重大得多，会发生什么呢？这个大权重会再次导致相应神经元的输出非常大。然后，这种不平衡将再次通过神经网络继续级联（向后传递），导致值较大的特征比值较小的特征值对学习过程的影响更大。

  That is the reason why **we need to normalize not just the input data, but also the data in the individual layers of the network.When applying batch norm to a layer we first normalize the output from the activation function.** 

  这就是为什么我们不仅需要规范化输入数据，还需要规范化网络各个层中的数据的原因。当对一个层应用batch norm时，我们首先将（上一层）激活函数的输出规范化。在对激活函数的输出进行规范化后，batch norm 将为每一层添加两个参数，参数可以在训练时进行学习。

  - 数据输入模型前，需要进行standard normalization，整个数据集的特征值的均值和标准差需要进行计算。
  - 数据输入模型后，需要进行batch normalization，一个mini-batch数据的特征值的均值和标准差需要进行计算。
  - 均值和方差计算得到的是一个标量，还是一个向量呢？（是向量）

- 过程：

  The larger data points in these non-normalized datasets can cause instability in neural networks because the relatively large inputs  cascade down through the layer in the network,which may cause imbalanced gradients,which may therefore cause famous exploding gradient problem. 

  What if ,during training,one of the weights ends up becoming drastically larger than the other weights? Well this large weight will then cause the output from its corresponding neuron to be extremely large,and this imbalance will,again,continue to cascade through the network,causing instability.**This is where batch Normalization comes into play.**

  When applying batch norm to a layer, the first thing batch norm does is normalize the output from the activation function.After normalizing the output from the activation function,batch norm multiplies this normalized output by some arbitrary parameter and then adds another arbitrary parameter to this resulting product.

  | Step | Expression      | Description                                            |
  | ---- | --------------- | ------------------------------------------------------ |
  | 1    | z = (x - μ) / σ | Normalize output x from activation function.           |
  | 2    | z * g           | Multiply normalized output z by arbitrary parameter g  |
  | 3    | (z * g) + b     | Add arbitrary parameter b to resulting product (z * g) |

- 计算过程示例：

  ![batch-normalization-example](D:\Typora\Notes\NLP\batch-normalization-example.png)

  - 计算细节

    - 在训练时，batch norm计算每个 mini-batch 的均值和标准差

    - 在推理时，不一定有一个batch可以计算均值和标准差，为了解决这个问题，模型在训练时可以维持一个移动平均的均值和标准差，这些值在训练时在所有batch上进行累计，在推理时用作均值和标准差。

      To overcome this limitation, the model works by maintaining a [moving average](https://mathworld.wolfram.com/MovingAverage.html) of the mean and variance at training time, called the moving mean and moving variance. These values are accumulated across batches at training time and used as mean and variance at inference time.

  - 计算公式
    $$
    x_{new} = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    $$
    

- batch norm 的局限性

  - In batch normalization, we use the batch statistics: the mean and standard deviation corresponding to the current mini-batch. However, when the batch size is small, the sample mean and sample standard deviation are not representative enough of the actual distribution and the network cannot learn anything meaningful.

    在batch norm中，我们是要batch统计信息，计算当前batch数据对应的均值和标准差，然而，当batch size 较小时，样本均值和标准差不足以代表实际分布，神经网络无法学习到任何有意义的东西。

  - As batch normalization depends on batch statistics for normalization, **it is less suited for sequence models**. This is because, in sequence models, we may have sequences of potentially different lengths and smaller batch sizes corresponding to longer sequences.

    由于batch norm 依赖于batch统计信息进行归一化，因此它不太适合序列模型。这是因为，在序列模型中，我们可能有潜在的不同长度的序列，以及较长的序列对应的batch size较小。

- Problems associated with Batch Normalization :

  1. **Variable Batch Size →** If batch size is of 1, then variance would be 0 which doesn’t allow batch norm to work. Furthermore, if we have small mini-batch size then it becomes too noisy and training might affect. There would also be a problem in **distributed training**. As, if you are computing in different machines then you have to take same batch size because otherwise γ and β will be different for different systems.

     首先，batch size 等于1时，方差为0此时batch norm无法进行计算；其次，如果batch size较小，会导致噪声太多，训练会受到影响；最后，在分布式训练情况下，不同的机器或显卡不得不取相同的batch size，否则 γ and β 将会变得不同。

  2. **Recurrent Neural Network** → In an RNN, the recurrent activations of each time-step will have a different story to tell(i.e. statistics). This means that we have to fit a separate batch norm layer for each time-step. This makes the model more complicated and space consuming because it forces us to store the statistics for each time-step during training.

### Layer Normalization

- 含义： In layer normalization, all neurons in a particular layer effectively have the same distribution across all features for a given input.Normalizing *across all features* but for each of the inputs to a specific layer removes the dependence on batches. **This makes layer normalization well suited for sequence models** such as Transformer and RNNs.

   Layer normalization normalizes input across the features instead of normalizing input features across the batch dimension in batch normalization.

- 计算过程

  ![layer-normalization](D:\Typora\Notes\NLP\layer-normalization.png)

- 计算公式：
  $$
  x_{new} = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
  $$

- 代码示例

  ```python
  import torch
  import torch.nn as nn
  
  batch, sentence_length, embedding_dim = 20, 5, 10
  embedding = torch.randn(batch, sentence_length, embedding_dim)
  # 在 embedding 维度计算10个特征的均值和方差，对每个time step的 hidden_state向量进行标准化
  layer_norm = nn.LayerNorm(embedding_dim)
  # Activate module
  norm_output = layer_norm(embedding)
  print(norm_output.size()) # torch.Size([20, 5, 10])
  ```

  内部实现

  ```python
  class LayerNorm(nn.Module):
      "Construct a layernorm module (See citation for details)."
      def __init__(self, feature_size, eps=1e-6):
          #初始化函数有两个参数，一个是features,表示词嵌入的维度,另一个是eps它是一个足够小的数，在规范化公式的分母中出现,防止分母为0，默认是1e-6。
          super(LayerNorm, self).__init__()
          #根据features的形状初始化两个参数张量a2，和b2，第一初始化为1张量，也就是里面的元素都是1，第二个初始化为0张量，也就是里面的元素都是0，这两个张量就是规范化层的参数。因为直接对上一层得到的结果做规范化公式计算，将改变结果的正常表征，因此就需要有参数作为调节因子，使其即能满足规范化要求，又能不改变针对目标的表征，最后使用nn.parameter封装，代表他们是模型的参数
          self.a_2 = nn.Parameter(torch.ones(feature_size))
          self.b_2 = nn.Parameter(torch.zeros(feature_size))
          #把eps传到类中
          self.eps = eps
  
      def forward(self, x):
      #输入参数x代表来自上一层的输出，在函数中，首先对输入变量x求其最后一个维度的均值，并保持输出维度与输入维度一致，接着再求最后一个维度的标准差，然后就是根据规范化公式，用x减去均值除以标准差获得规范化的结果。
      #最后对结果乘以我们的缩放参数，即a2,*号代表同型点乘，即对应位置进行乘法操作，加上位移参b2，返回即可
          mean = x.mean(-1, keepdim=True)
          std = x.std(-1, keepdim=True)
          return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
  ```
  
  
  
  

### Batch norm vs. Layer norm

- Batch normalization normalizes each feature independently across the mini-batch. Layer normalization normalizes each of the inputs in the batch independently across all features.

  batch norm 保持了特征的独立性，对每个样本的某个特征计算均值和标准差。

  layer norm 保持了样本的独立性，对每个样本的所有特征值计算均值和标准差。

- As batch normalization is dependent on batch size, it’s not effective for small batch sizes. Layer normalization is independent of the batch size, so it can be applied to batches with smaller sizes as well.

  batch size 对batch norm的作用有影响；而batch size 对 layer norm 没有影响，

- Batch normalization requires different processing at training and inference times. As layer normalization is done along the length of input to a specific layer, the same set of operations can be used at both training and inference times.

​       batch norm在训练和推理时的处理不同，而layer norm在训练和推理时的处理相同。



