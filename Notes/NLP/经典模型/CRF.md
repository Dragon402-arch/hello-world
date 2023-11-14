## Conditional Random Fields

[三大经典模型完整教程](http://www.cs.columbia.edu/~mcollins/crf.pdf)

### 1. MEMM 模型

#### 1.1 MEMM模型基础

MEMM（Maximum-Entropy Markov model）即最大熵马尔科夫模型，**该模型是判别式模型，也是有向图模型**，建模条件概率分布：
$$
p(s_1,s_2,. . .,s_m|x_1,x_2,. . .,x_m)
$$
其中 $x_i$ 是观测值，$s_i$ 是状态值。

![MEMM](D:\Typora\Notes\NLP\经典模型\MEMM.png)

一旦定义了特征向量 $\phi$，通过最大化似然函数，就可以得到模型参数 $w$ ，训练数据由观测序列 $x_1,x_2,. . .,x_m$ 和状态序列 $s_1,s_2,. . .,s_m$组成。一旦有了训练的模型参数 $w$，就有了模型：
$$
p(s_i|s_{i-1},x_1,x_2,. . .,x_m)
$$
选取概率最大的状态取值，从而得到了状态转移结果，进而可以计算：
$$
p(s_1,s_2,. . .,s_m|x_1,x_2,. . .,x_m)
$$
模型的继承：

- MEMM模型也引入了一阶马尔科夫假设，即当前时刻的状态取值只由前一个时刻的状态取值决定。

模型的改进：

- MEMM模型建模条件概率分布，去除了HMM中观测值只依赖于当前时刻状态值的假设
- 解决了HMM模型存在的目标函数与预测解码函数并不一致的问题
- MEMM模型使用了特征向量  $\phi$，允许引入更加丰富的特征。
- 在给定当前时刻状态取值的情况下建模下一个时刻状态的分布，也就是使用softmax函数进行状态多分类，取概率最大的状态作为转移结果。因此下一个时刻的状态取值不仅取决于当前时刻的状态取值，也取决于模型的权重参数。MEMM is not a generative model, but a model with finite states based on state classification，**defining the state distribution of the next state under the current state conditions given**.

存在的问题：

- 存在标签偏置（label bias）问题。

  **MEMM tends to select the state with fewer convertible states.** **MEMM biased towards states with few successor states.** Such selection is termed the labeling bias issue. CRF well addresses the labeling bias issue.

  问题出现原因：由于MEMM选取局部状态特征进行归一化，状态转移时追求局部最优，使得无法得到全局最优解。

#### 1.2 标签偏置问题

**MEMM tends to select the state with fewer convertible states.** **MEMM biased towards states with few successor states.** Such selection is termed the labeling bias issue. CRF well addresses the labeling bias issue.

MEMM模型倾向于选择具有较少可转换状态的状态，这种选择倾向被称为标签偏置问题。

**One of the motivations of conditional random fields was to avoid the label-bias problem found in hidden Markov models and maximum entropy Markov models(MEMM)). MEMMs are like conditional random fields, but normalize their probabilities on a per-tag basis rather than over the whole sequence.**

条件随机场的动机之一是避免隐马尔可夫模型HMM和最大熵马尔可夫模型MEMM中存在的标签偏置问题。MEMM类似于条件随机场，但是在每个标签的基础上归一化它们的概率，而不是在整个序列上。

### 2. CRF 模型

**学习资源**： [HMM、MEMM、CRF对比分析](https://alibaba-cloud.medium.com/hmm-memm-and-crf-a-comparative-analysis-of-statistical-modeling-methods-49fc32a73586)、[CRF讲解](https://medium.com/ml2vec/overview-of-conditional-random-fields-68a2a20fa541)、[CRF代码朴素实现](https://towardsdatascience.com/conditional-random-field-tutorial-in-pytorch-ca0d04499463)、[讲解2](https://www.depends-on-the-definition.com/named-entity-recognition-conditional-random-fields-python/)、[讲解3](http://www.cs.columbia.edu/~mcollins/)

理解关键

1. **The CRF model has addressed the labeling bias issue and eliminated two unreasonable hypotheses in HMM. CRF computes the joint probability distribution of the entire label sequence when an observation sequence intended for labeling is available, rather than defining the state distribution of the next state under the current state conditions given.**

   **在给定观测序列的情况下，CRF计算的是整个状态序列的联合概率分布，而不是在给定的当前状态的条件下建模下一个时刻状态的状态分布。**

2. 标签约束是通过权重取值（状态转移概率）来实现的，权重取值来自模型训练结果。如果要求一个标签后边不能另一个标签，则对应的参数值在模型训练后取值就会比较小，反之就会取值比较大。此外CRF中的参数值（状态转移概率）是没有标准化的概率取值，也就是一行的值相加结果不等于1.

3. 通过建模整个状态序列的联合概率分布，而不是建模单个状态的分布，从而解决了标签偏置问题（倾向于转移到具有较少后继状态的状态）

4. 相比于HMM，去除了两个不合理的独立性假设。

CRF（Conditional Random Fields）即条件随机场，**该模型是判别式模型，也是无向图模型**，建模条件概率分布：
$$
p(s_1,s_2,. . .,s_m|x_1,x_2,. . .,x_m) = P(S|X)
$$
定义全局特征向量（global feature vector）：
$$
\Phi(X,S)=\sum_{j=1}^m\phi(X,s_{j-1},s_j,j) \in R^d
$$
全局特征向量的意思在于其将整个状态序列考虑在内。其中 $\phi(·)\in R^d$ ，与MEMM模型中的特征向量相同，。

定义 CRF 模型并展开：
$$
\begin{align} P(S|X,\vec w) &= \large \frac{exp(\vec w · \Phi(X,S)  )}{\sum_{S^{'}\in S^m}exp(\vec w · \Phi(X,S^{'}) } \\
&= \large \frac{1}{Z} exp(\vec w · \Phi(X,S)) \\
&= \large \frac{1}{Z} exp(\vec w ·\sum_{j=1}^m\phi(X,s_{j-1},s_j,j))  \\
&= \large\frac{1}{Z} exp(\sum_{j=1}^m\vec w ·\phi(X,s_{j-1},s_j,j))  \\
&= \large \frac{1}{Z} exp(\sum_{j=1}^m\sum_{i=1}^d w_i \phi_i(X,s_{j-1},s_j,j))  \\
&= \large \frac{1}{Z} exp\bigg[\sum_{j=1}^m\sum_{i=1}^d w_i \bigg(f_i(X,s_{j-1},s_j,j)+g_i(X,s_j,j)\bigg)\bigg]  \\
&= \large \frac{1}{Z} exp\bigg[\sum_{j=1}^m\sum_{i=1}^d \bigg(\lambda_if_i(X,s_{j-1},s_j,j)+\mu_ig_i(X,s_j,j)\bigg)\bigg]  \\
\end {align}
$$
其中 $\vec w \in R^d$ 是模型的参数，$\vec w\ ·\phi(X,s_{j-1},s_j,j) $ 表示内积运算，结果是一个标量，表示状态变量从 $s_{j-1}$ 转移到 $s_j$ 的相关得分，其取值可正可负；其中 $\phi_i(X,s_{j-1},s_j,j)$ 表示取 $d$ 维向量的第 $i$ 个元素。



### 3. HMM与CRF关系

结论：**每个HMM模型都对应一个特定的CRF模型，但是HMM不是CRF的特例，因为HMM建模的是联合概率分布（HMM），从该联合概率分布可以导出一个条件概率分布（CRF）**。

推导：
$$
\begin{aligned} P(X,Y) &= P(x_1,…,x_T,y_1,…,y_T)\\
  & =P(y_1)P(x_1|y_1)\prod_{t=2}^{T}P(x_t|y_t)P(y_t|y_{t-1}) \\
  & = \prod_{t=1}^{T}P(x_t|y_t)P(y_t|y_{t-1}) \\
  & =exp\bigg\{logP(X,Y))\bigg\} \\
  & = exp \bigg\{ \sum_{t} log\big(P(y_t|y_{t-1})\big) + \sum_{t} log\big(P(x_t|y_t)\big)\bigg\} \end{aligned}
$$
  为了简化表示，将initial state distribution初始概率分布 $$P(y_1)$$ 写为$$P(y_1|y_0)$$.

  引入指示函数 $$I(·)$$，为真则取1，为假则取0。
$$
  P(X,Y)= exp \bigg\{\sum_{t}\sum_{i,j∈S} log\big(P(y_t=j|y_{t-1}=i)I(y_{t-1}=i)I(y_{t}=j)\big) +   
  \sum_{t}\sum_{i∈S}\sum_{o∈O} log\big(P(x_t=o|y_t=i)I(y_t=i)I(x_t=o)\big) \bigg\}
$$
  令 $$λ_{ij}=log\big(P(y_t=j|y_{t-1}=i)$$，$$\mu_{oi}=log\big(P(x_t=o|y_t=i)\big)$$,则上述公式可变形为：
$$
P(X,Y) = exp \bigg\{\sum_{t}\sum_{i,j∈S} λ_{ij}I(y_{t-1}=i)I(y_{t}=j)\big) +   
  \sum_{t}\sum_{i∈S}\sum_{o∈O}\mu_{oi}I(y_t=i)I(x_t=o)\big) \bigg\}
$$
  若 $$λ_{ij},\mu_{oi}$$ 不再等于上面的值（等于更一般的值），引入归一化常数Z，使得概率和为1，则可以将该等式更一般地推广为：
$$
  P(X,Y)=\frac{1}{Z} exp \bigg\{\sum_{t}\sum_{i,j∈S} λ_{ij}I(y_{t-1}=i)I(y_{t}=j)\big) +   
  \sum_{t}\sum_{i∈S}\sum_{o∈O}\mu_{oi}I(y_t=i)I(x_t=o)\big) \bigg\}
$$
  若引入特征函数，$$f_{ij} (y,y',x) =I(y=i)I(y'=j),f_{io} (y,y',x) =I(y=i)I(x=o) $$,则上式可改写为，此时的Z为归一化常数
$$
  P(X,Y)=\frac{1}{Z} exp \bigg\{\sum_{t}\sum_{k} w_{k}f_k(y_t,y_{t-1},x_t) \bigg\}
$$
  条件概率分布为：
$$
P(Y|X) =\frac{P(X,Y)}{P(X)}=\frac{P(X,Y)} {{\sum_{Y'}}{P(X,Y')}} \\
  =\frac{exp \bigg\{\sum_{t}\sum_{k} w_{k}f_k(y_t,y_{t-1},x_t) \bigg\}}{{\sum_{y'}}{exp \bigg\{\sum_{t}\sum_{k} w_{k}f_k(y'_t,y'_{t-1},x_t) \bigg\}}}
$$

可以看到该条件分布是一个特殊的线性链条件随机场，且在每个时刻特征函数在观测序列输入方面只有当前时刻的观测值，而一般的CRF在每个时刻都是将观测序列作为输入的。

**This conditional distribution is a linear-chain CRF, in particular one that includes features only for the current word’s identity. But many other linear-chain CRFs use richer features of the input, such as prefixes and suffixes of the current word, the identity of surrounding words, and so on.**






### 4. HMM、MEMM、CRF对比

HMM缺陷：

- 一阶马尔科夫假设不符合实际情况，序列标注标注问题中还依赖上下文等特征。
- 目标函数与预测（解码）函数并不一致，目标函数时建模观测序列和状态序列的联合概率分布$P(X,Y)$ ，而预测函数则使用的是条件概率分布$P(Y|X)$。
- 存在标签偏置（labeling bias）问题。

MEMM模型改进：

- 建模条件概率分布，解决了HMM模型存在的目标函数与预测函数并不一致的问题，也不必考虑HMM中的发射概率部分。
- 考虑了相邻状态变量 $ s_i,s_{i+1}$ 与整个观测序列 $X$ 之间依赖关系，而HMM则是只考虑了相邻状态变量 $ s_i,s_{i+1}$ 与当前观测值 $x_{t+1}$ 之间依赖关系

MEMM模型缺陷：

- 存在标签偏置（labeling bias）问题。

CRF模型改进：

- 相比于HMM，去除了两个不合理的独立性假设

- 相比于MEMM，使用全局特征进行归一化，解决了标签偏置问题。

- **在给定观测序列的情况下，CRF计算的是整个状态序列的联合概率分布，而不是在给定的当前状态的条件下定义下一个时刻状态的状态分布。**

  

### 5. 生成式模型与判别式模型

#### 5.1 生成式模型

- 模型特征： Infinite samples --> Probability density model = Generative model --> Prediction
- 训练数据：需要无限数据样本或是尽可能多的数据样本

- 代表模型：NaiveBayes、HMM，GMM，LDA（隐狄利克雷分布）

- 建模目标：建模联合概率分布，建模数据是如何生成的。

- 建模示例：以朴素贝叶斯模型为例：

  首先建模联合概率分布 $P(X,y)$ ，然后使用联合概率分布计算条件概率分布 $P(y|X)$，并依据属性条件独立性假设进行展开 。
  $$
  P(y|X) = \frac{P(X,y)}{P(X)}= \frac{P(y)P(X|y)}{P(X)} = \frac{P(y)}{P(X)} \prod_{i=1}^dP(x_i|y)
  $$
  其中 $P(X|y)$ 是似然，$P(y)$ 是先验概率，$P(y)$ 表达了样本空间中各类别样本所占的比例。
  $$
  y^* = \arg \limits \max \limits_{y} P(y) \prod_{i=1}^dP(x_i|y)
  $$

#### 5.2 判别式模型

- 模型特征：Finite samples --> Discriminative function = Discriminative model --> Prediction
- 训练数据：需要有限数据样本，相比于生成式模型所需数据样本是较少的。

- 代表模型：逻辑回归（也被称为最大熵分类器），决策树，SVM，CRF、MEMM

- 建模目标：建模条件概率分布，建模不同类别样本点之间的决策边界。

- 建模示例：以逻辑回归模型为例：
  $$
  y^* = \arg \limits \max \limits_{y} P(y|X) 
  $$
  











