## 1. XGBoost的原理推导

[官方教程](https://xgboost.readthedocs.io/en/latest/tutorials/model.html)

[讲解](https://mp.weixin.qq.com/s/YunDfYPLywc0tMJF72YIAQ)

XGBoost（eXtreme Gradient Boosting）其实是GBDT算法在工程上的一种实现方式。 

**Is XGBoost and gradient boosting same?**

Yes, XGBoost (Extreme Gradient Boosting) is a specific implementation of gradient boosting. While gradient boosting is a general ensemble learning method that combines multiple weak learners (e.g., decision trees) to create a strong predictor, **XGBoost is a highly optimized and efficient implementation of gradient boosting that incorporates additional regularization techniques and speed enhancements, making it widely popular in machine learning competitions and real-world applications.**

##### 2.2.3 LightGBM

 加法模型（Additive Training）：
$$
\begin{aligned}  \hat y_i^{(0)} &= 0 \\
\hat y_i^{(1)} &= f_1(x_i) =  \hat y_i^{(0)}  + f_1(x_i) \\
\hat y_i^{(2)} &= f_1(x_i) + f_2(x_i) =  \hat y_i^{(1)}  + f_2(x_i) \\
… \\
\hat y_i^{(t)} &= \sum_{k=1}^tf_k(x_i) =  \hat y_i^{(t-1)}  + f_t(x_i) \\
\end {aligned}
$$
优化目标：
$$
\begin{aligned}   obj^{(t)} &= \sum_{i=1}^nL(y_i,\hat y_i^{(t)}) + \sum_{k=1}^t \Omega(f_i) \\
			&= \sum_{i=1}^nL(y_i,\hat y_i^{(t-1)}+f_t(x_i)) + \Omega(f_t)+constant
\end {aligned}
$$
在使用MSE作为损失函数的情况下，目标函数可以转换为：
$$
\begin{aligned}   obj^{(t)} &= \sum_{i=1}^nL(y_i,\hat y_i^{(t-1)}+f_t(x_i)) + \Omega(f_t)+constant \\
&= \sum_{i=1}^n[(y_i-(\hat y_i^{(t-1)}+f_t(x_i))^2] + \Omega(f_t)+constant \\
&= \sum_{i=1}^n[((y_i-\hat y_i^{(t-1)})-f_t(x_i))^2] + \Omega(f_t)+constant \\
&= \sum_{i=1}^n[2(\hat y_i^{(t-1)}-y_i)f_t(x_i)+f_t^2(x_i)] + \Omega(f_t)+constant \\
\end {aligned}
$$
在不限定损失函数的情况下，也就是一般情况下，可以将损失函数使用泰勒公式进行展开：

泰勒公式：
$$
f(x+\Delta x)\approx   f(x) + f'(x)\Delta x +   \frac{1}{2} f''(x)\Delta x
$$
可以将 $L(y_i,\hat y_i^{(t-1)}+f_t(x_i))$ 中的 $\hat y_i^{(t-1)}$ 看作 $x$ ，将 $f_t(x_i)$ 看作 $\Delta x$ ，则有：
$$
\begin{aligned}   obj^{(t)} &= \sum_{i=1}^nL(y_i,\hat y_i^{(t-1)}+f_t(x_i)) + \Omega(f_t)+constant \\
&\approx  \sum_{i=1}^n[L(y_i,\hat y_i^{(t-1)})+g_if_t(x_i)+ \frac{1}{2} h_if_t^2(x_i)] + \Omega(f_t)+constant \\
&=  \sum_{i=1}^n[g_if_t(x_i)+ \frac{1}{2} h_if_t^2(x_i)] + \Omega(f_t)+constant
\end {aligned}
$$
其中 $g_i$ 和 $h_i$ 定义为： 
$$
g_i=\frac{\partial L(y_i,\hat y_i^{(t-1)})}{{\partial \hat y_i^{(t-1)}}} \\
h_i=\frac{\partial^2 L(y_i,\hat y_i^{(t-1)})}{{\partial \hat y_i^{(t-1)}}}
$$ { }
在去除常数项后，第 $t$ 棵树的目标函数变为：
$$
\sum_{i=1}^n[g_if_t(x_i)+ \frac{1}{2} h_if_t^2(x_i)] + \Omega(f_t)
$$
通过引入正则项来控制模型复杂度：

We need to define the complexity of the tree $\Omega(f_t)$. In order to do so, let us first refine the definition of the tree $f_t(x)$ as
$$
f_t(x) = w_{q(x)}， w \in R^T, q:R^d \rightarrow {1,2,…,T}
$$
Here $w$ is the vector of scores on leaves, $q(x)$ is a function assigning each data point to the corresponding leaf, and $T$ is the number of leaves.  $w$ 是所有叶结点得分取值构成的向量，$T$ 表示第 $t$ 棵树叶结点的个数，$d$ 表示输入 $x$ 特征的个数，而如 $q(x_3) = 4$ 的意思则表示将 $x_3$ 分配到了第 4 个叶结点，而分配给该叶结点的取值为 $w_4$ 。

In XGBoost, we define the complexity as（定义树的复杂度为）：
$$
\Omega(f_t) = \gamma T + \frac{\lambda}{2} \sum_{j=1}^Tw_j^2
$$
将模型复杂度表示和第 $t$ 棵树表示带入目标函数后可得：
$$
\begin{aligned}   obj^{(t)} &= \large \sum_{i=1}^n[g_if_t(x_i)+ \frac{1}{2} h_if_t^2(x_i)] + \Omega(f_t) \\
&= \large \sum_{i=1}^n[g_iw_{q(x_i)}+ \frac{1}{2} h_iw_{q(x_i)}^2] + \gamma T + \frac{\lambda}{2} \sum_{j=1}^Tw_j^2 \\
&= \large \sum_{j=1}^n[(\sum_{i \in I_j}g_i)w_j+ \frac{1}{2} (\sum_{i \in I_j}h_i+\lambda )w_j^2] + \gamma T  \\
&= \large \sum_{j=1}^n[G_jw_j+ \frac{1}{2} (H_j+\lambda )w_j^2] + \gamma T  \\
\end {aligned}
$$
其中  $I_j = \{i|q(x_i)=j\}$，表示分配给第 $j$ 个叶结点的样本索引集合，如 $x_1,x_5,x_9$ 分配给了第 $j$ 个叶结点，则 $I_j=\{1,5,9\}$。第二行到第三行的的变换是做了从遍历样本到遍历树的叶结点。对于一个给定的树结构 $q(x)$ （也就是每个叶结点的样本集合是确定的情况下），在上面的等式中 $w_j$ 可看做变量，其余都可以看作常量，实际上就是一个开口向上的二次函数，其在 $-\frac{b}{2a}$ 处取得最小值，可以计算叶结点 $j$ 的最优得分为 $w^*_j$为以及相应的最优目标函数值为：
$$
w_j^*= -\frac{G_j}{H_j+ \lambda} \\
obj^*=-\frac{1}{2} \sum_{j=1}^T \frac{G_j^2}{H_j+ \lambda} +  \gamma T
$$
目标函数值越小，表示这个树的结构就越好。根据上面的公式，可以计算分别计算每个叶结点的最优输出值（作为对比，最小二乘回归树中每个叶结点的最优输出值为叶结点样本集合的均值）。

## 2. 一棵树的生成细节

### 2.1 特征选择准则

现在我们有了衡量一棵树好坏的方法，理想情况下，我们应该列举所有可能的树，然后选出最好的树。在实践中，这是很难的，所以我们将尝试一次优化树的一个level（也就是一层）。具体而言，我们试图将一个叶节点分裂称为左右两个叶结点，产生的增益为：
$$
\begin{aligned}  Gain &= obj^{Before} - obj^{After}\\
&= -\frac{1}{2}  \frac{G^2}{H+ \lambda} +  \gamma  - [-\frac{1}{2}(\frac{G_L^2}{H_L+\lambda} +  \frac{G_R^2}{H_R+\lambda})+2\gamma ] \\
&= \frac{1}{2} \bigg[ \frac{G_L^2}{H_L+\lambda} +  \frac{G_R^2}{H_R+\lambda} -  \frac{G^2}{H +\lambda}\bigg] -\gamma  \\
&=  \frac{1}{2} \bigg[ \frac{G_L^2}{H_L+\lambda} +  \frac{G_R^2}{H_R+\lambda} -  \frac{(G_L+G_R)^2}{H_L+H_R+\lambda}\bigg] -\gamma 
\end {aligned}
$$
该特征收益也可以作为特征重要性输出的重要依据。

This formula can be decomposed as ：

1) the score on the new left leaf 
2)  the score on the new right leaf 
3) The score on the original leaf 
4) regularization on the additional leaf. 

We can see an important fact here: if the gain is smaller than $\gamma$, we would do better not to add that branch. This is exactly the **pruning** techniques in tree based models!

### 2.2 确定树的结构

为了便于理解，一棵树的学习过程可以大致描述为：

1. 枚举所有可能的树结构
2. 为每个树结构计算其目标函数值，值越小说明对应的树结构越好
3. 根据上一步的结果，找到最佳的树结构，并为树的每个叶子节点计算预测值

然而，可能的树结构数量是无穷的，所以实际上我们不可能枚举所有可能的树结构。通常情况下，我们采用贪心策略来生成一棵树的每个结点。

#### 2.2.1 贪心算法

思想：分布、局部最优（贪恋），每次只计算一个结点的最优划分，按照增益来决定是否进行划分。

Exact Greedy Algorithm：a split finding algorithm enumerates over all the possible splits on all the features.

1. 从深度为0的树开始，对每个叶节点枚举所有的可用特征。
2. 针对每个特征，把属于该结点的训练样本按照该特征值进行升序排列，通过线性扫描的方式来确定该特征的**最佳分裂点**，并记录该特征的最大分裂收益（采用最佳分裂点时的收益）；这里假设类别型特征已经做了one-hot编码。
3. 选择分裂收益最大的特征作为**分裂特征**，用该特征的最佳分裂点作为分裂位置，将该结点分裂出左右两个新的叶结点，并为每个新结点关联对应的样本集合。
4. 回到第1步，递归执行到满足特定条件为止。

在上述算法的第二步，样本排序的时间复杂度为$O(n\log n)$，假设共有 *K* 个特征，那么生成一颗深度为 *d* 的树的时间复杂度为 $O(dKn\log n)$。具体实现可以进一步优化计算复杂度，比如可以缓存每个特征的排序结果等。（其中 d 表示样本总特征数，m 表示特征采样后的总特征数）

![贪心算法和近似算法](D:\Typora\Notes\NLP\经典模型\贪心算法和近似算法.png)

贪心算法得到的结果比较精确，但操作较为耗时，从上面的伪代码中可以看到其包含内外两层循环以及排序操作，有以下优化方向：

- 外层循环：对于样本的每个特征都要进行排序和内循环操作，因此可以考虑进行特征采样优化，比如八个特征中采样到五个特征。
- 内层循环：对于每个特征值都作为候选最佳分裂点，都需要计算增益值，因此可以考虑只选取部分特征值，比如选择分位数点的特征值。
- 排序操作

#### 2.2.2 近似算法

思想：优化贪心算法，进行特征采样和降低候选最佳分裂点的数量。

##### 2.2.2.1 采样

每次训练时，对数据集采样，可以增加树的多样性，降低模型过拟合的风险。另外，对数据集采样还能减少计算，加快模型的训练速度。**在降低过拟合风险中，进行特征采样比进行样本采样的效果更显著。**

###### 样本采样

默认是 $default=1$ 不进行样本采样。样本采样方式：

- 认为每个样本平等水平，对样本集进行相同概率采样；
- 认为每个样本是不平等的，每个样本对应的一阶、二阶导数信息表示优先级，导数信息越大的样本越有可能被采到。

![样本采样](D:\Typora\Notes\NLP\经典模型\样本采样.png)

###### 特征采样

特征采样，又称为列采样（column subsampling）：

![列采样](D:\Typora\Notes\NLP\经典模型\列采样.png)

特征采样方式有三种，第一种是在构建每棵树时进行特征采样；第二种特征采样范围是在第一种的基础上，对于树的每一层级（树的深度）进行特征采样；第三种特征采样范围是在第二种的基础上，对于每个树节点进行特征采样。这三种特征采样方式有串行效果。比如，当第一、二、三种的特征采样比例均是0.5时，如果特征总量为64个，经过这三种采样的综合效果，最终采样得到的特征个数为8个。

**也就是先树采样，然后树采样的基础上进行层级采样，然后在层级采样的基础上进行结点采样。**

##### 2.2.2.2 特征值分桶

​	虽然贪心算法可以得到最优解，但当数据量太大时则无法读入内存进行计算，近似算法（Approximate Algorithm）主要针对贪心算法这一缺点给出了近似最优解。对于每个特征，只考察分位数点可以降低计算量。该算法首先根据样本的特征分布的分位数提出候选划分点，然后将连续型特征映射到由这些候选点划分的桶中，然后聚合统计信息找到所有区间的最佳分裂点。（使用分位数进行数据分桶）

简单来讲，近似算法首先取特征 $k$ 的 $l$ 个分位数点作为候选切分点 $S_k= \{s_{k1},s_{k2},…,s_{kl}\}$，从而不必再遍历该特征的所有特征取值，其中获取特征候选切分点的策略有Local 和 Global 两种。然后将样本按照特征$k$ 的取值映射到对应的桶中（是一个区间，即$s_{k,v}\ge x_{jk}>s_{k,v-1}$），对每个桶内样本的 $G,H$ 进行累加，最后在候选切分点集合上贪心查找。

在提出候选切分点时有两种策略：

- Global：The global variant proposes all the candidate splits during the initial phase of tree construction, and uses the same proposals for split finding at all levels.

  针对一棵树，只在树构建前进行一次数据分桶操作，在后续树的构建过程中使用最初的分桶结果。

- Local：The local variant re-proposes after each split. 

  在每个结点需要分裂时都重新进行一次数据分桶操作，此时分桶操作只限定在该结点所包含的样本中。

- 直观上来看，Local策略需要更多的计算步骤，而Global策略因为节点已有划分所以需要更多的候选点。

  ![Globa and Local](D:\Typora\Notes\NLP\经典模型\Globa and Local.png)

As you can see from the figure above, when it is Global Variant, parent node will split into two child nodes from the best split point. As you can see from the red lines representing the bucket, it is maintaining the parent node’s bucket points.

Local variant on the other hand, does the bucketing process every time there is a split from the parent node. Here the two child nodes (left and right) will have 10 buckets again.

**For global variant since it is maintaining the bucket, it would be best to get as many bucket as possible. Local variant, since there is bucketing every time there is a split, it is beneficial on the bigger tree.**Bucketing is one of the hyperparameter of XGBoost. Eps is a hyperparameter that defines the bucket number.1/Eps = Number of Candidate points，比如 $1 / 0.1 = 10 $，表示有10个桶。

计算示例如下：

![近似算法分位数](D:\Typora\Notes\NLP\经典模型\近似算法分位数.jpg)

首先对样本特征值进行升序排列，然后使用三分位数可以得到2个候选划分点（两个分位点以及不切分），最后分别计算四种划分情况的增益值。

##### 2.2.2.3 加权分位数

实际上，XGBoost不是简单地按照样本个数来获取分位数的，而是以样本的二阶导数值 $h_i$ 作为样本的权重来划分分位数的。示例数据如下：

![加权分位数](D:\Typora\Notes\NLP\经典模型\加权分位数.png)

首先按照样本特征值进行升序排列，至于 $h_i$ 则不一定是升序排列的，上图的情况只是特殊情况。

使用二阶导数值 $h_i$ 作为样本权重值的原因在于：
$$
\begin{aligned}   obj^{(t)} &= \large \sum_{i=1}^n[g_if_t(x_i)+ \frac{1}{2} h_if_t^2(x_i)] + \Omega(f_t) \\
&= \large \sum_{i=1}^n[ \frac{1}{2} h_i\bigg(\frac{2g_if_t(x_i)}{h_i}+f_t^2(x_i)\bigg)] + \Omega(f_t) \\
&= \large \sum_{i=1}^n[ \frac{1}{2} h_i\bigg(\frac{g_i}{h_i}+f_t(x_i)\bigg)^2] + \Omega(f_t) - \sum_{i=1}^n\frac{g_i^2}{2h_i}\\
&= \large \sum_{i=1}^n[ \frac{1}{2} h_i\bigg(\frac{g_i}{h_i}+f_t(x_i)\bigg)^2] + \Omega(f_t) - constant \\
&= \large \sum_{i=1}^n[ \frac{1}{2} h_i\bigg(f_t(x_i)-(-\frac{g_i}{h_i})\bigg)^2] + \Omega(f_t) - constant \\
\end {aligned}
$$
which is exactly weighted squared loss with labels $g_i/h_i$ and weights $h_i$. 

可以看到 $h_i$​ 就是平方损失函数中样本的权重。回归问题中使用MSE损失函数时 $h_i=1$  。

切分点$\{s_{k1},s_{k2},…,s_{kl}\}$ 应满足：
$$
|r_k(s_{k,j})-r_k(s_{k,j+1})| < \epsilon  \ s_{k1}=  \min \limits_i x_{ik},\ s_{kl}=  \max \limits_i x_{ik}
$$
其中 $k$ 表示样本的第 $k$ 个特征，$j = 1,2,3,…,l$，其中 $r_k(z)$ 的定义如下：
$$
r_k(z) = \frac{1}{\sum_{(x,h) \in D_k}h}\sum_{(x,h) \in D_k,x<z} h
$$
比如 $\epsilon =\frac{10}{27},s_{k1} = \frac{1}{27},s_{k2}=\frac{10}{27}，s_{k3}=\frac{19}{27}$,就可以满足上述等式。

## 3. 稀疏感知算法

实际工程中一般会出现输入值稀疏的情况。比如数据的缺失、one-hot 编码都会造成输入数据稀疏。XGBoost在构建树的节点过程中**只考虑非缺失值的数据遍历**，而为每个节点增加了一个缺省方向（也就是为特征值缺失的样本分配到左结点还是右节点，确定一个结点），当样本相应的特征值缺失时，可以被归类到缺省方向上，最优的缺省方向可以从数据中学到。至于如何学到缺省值的分支，其实很简单，**分别枚举特征缺省的样本归为左右分支后的增益，选择增益最大的枚举项即为最优缺省方向。**（直接将所有缺失值样本（看做整体）分配在左侧，分配在右侧分别算一下，那个增益高就放在那边，逐个缺失值样本遍历效率较低）

在构建树的过程中需要枚举特征缺失的样本，乍一看这个算法会多出相当于一倍的计算量，但其实不是的。因为在算法的迭代中只考虑了非缺失值数据的遍历，缺失值数据直接被分配到左右节点，所需要遍历的样本量大大减小。

**shrinkage**

在XGBoost中也加入了步长 $\eta$，也称为收缩率：
$$
\large \hat y_i^{(t)} = \sum_{k=1}^t\eta_kf_t(x_i) =  \hat y_i^{(t-1)}  + \eta_tf_t(x_i)
$$
这有助于防止过拟合，步长 $\eta$ 通常取 $0.1$ 。

## 4. 算法工程实现方面优化

### 4.1  列块并行学习

特征并行。

### 4.2 缓存访问

Each computer has CPU and which has a small amount of Cache memory. The CPU can use this memory faster than any other memory in computer. It allocates an internal buffer in each thread, to fetch gradient statistics into it, and it performs accumulation in mini-bath manner. **This can reduce the runtime when dataset is big.**

### 4.3  “核外”块计算

- 块压缩（Block Compression）

  Data is divided into multiple blocks and stored into disks. These blocks are compressed by columns and decompressed by independent threads when loading into main memory. This reduces the disk reading cost.

- 块分区（Block Sharding ）

  

## 5. 常见问题：

- XGBoost为什么需要对目标函数进行泰勒展开？
  1. 对目标函数进行泰勒展开，就是为了统一目标函数的形式，针对回归和分类问题，使得平方损失或逻辑损失函数优化求解，可以共用同一套算法框架及工程代码。
  2. 对目标函数进行泰勒展开，可以使得XGBoost支持自定义损失函数，只需要新的损失函数二阶可导即可，从而**提升算法框架的扩展性**。
  3. 相对于GBDT的一阶泰勒展开，XGBoost采用二阶泰勒展开，可以更精准的逼近真实的损失函数，**提升算法框架的精准性**。
  4. 一阶导数描述梯度的变化方向，二阶导数可以描述梯度变化方向是如何变化的，利用二阶导数信息更容易找到极值点。因此，基于二阶导数信息能够让梯度收敛的更快，类似于牛顿法比SGD收敛更快。

- XGBoost如何进行采样？

  XGBoost算法框架，参考随机森林的Bagging方法，支持样本采样和特征采样。由于XGBoost里没有交代是有放回采样，认为这里的样本采样和特征采样都是无放回采样。每次训练时，对数据集采样，可以增加树的多样性，降低模型过拟合的风险。另外，对数据集采样还能减少计算，加快模型的训练速度。在降低过拟合风险中，对特征采样比对样本采样的效果更显著。

  **样本采样**（如图7所示），默认是 $default=1$ 不进行样本采样。样本的采样的方式有两种，一种是认为每个样本平等水平，对样本集进行相同概率采样；另外一种认为每个样本是不平等，每个样本对应的一阶、二阶导数信息表示优先级，导数信息越大的样本越有可能被采到。

  **特征采样**（如图8所示），默认 $default=1$ 对特征不进行采样。对特征的采样方式有三种，第一种是在建立每棵树时进行特征采样；第二种特征采样范围是在第一种的基础上，对于树的每一层级（树的深度）进行特征采样；第三种特征采样范围是在第二种的基础上，对于每个树节点进行特征采样。这三种特征采样方式有串行效果。比如，当第一、二、三种的特征采样比例均是0.5时，如果特征总量为64个，经过这三种采样的综合效果，最终采样得到的特征个数为8个。

- 

[问题总结](https://mp.weixin.qq.com/s/YunDfYPLywc0tMJF72YIAQ)

- 与GBDT相比，Xgboost的优化点：

1. - 算法本身的优化：首先GBDT只支持决策树，Xgboost除了支持决策树，可以支持多种弱学习器，可以是默认的gbtree, 也就是CART决策树，还可以是线性弱学习器gblinear以及DART；其次GBDT损失函数化简的时候进行的是一阶泰勒公式的展开，而Xgboost使用的是二阶泰勒公式的展示。还有一点是Xgboost的目标函数加上了正则项，这个正则项是对树复杂度的控制，防止过拟合。
   - 可以处理缺失值。尝试通过枚举所有缺失值在当前节点是进入左子树，还是进入右子树更优来决定一个处理缺失值默认的方向
   - 运行效率：并行化，单个弱学习器最耗时的就是决策树的分裂过程，对于不同特征的特征分裂点，可以使用多线程并行选择。这里想提一下，我自己理解，这里应该针对的是每个节点，而不是每个弱学习器。这里其实我当时深究了一下，有点混乱。为什么是针对每个节点呢？因为我每个节点也有很多特征，所以在每个节点上，我并行（多线程）除了多个特征，每个线程都在做寻找增益最大的分割点。还有需要注意的一点是Xgboost在并行处理之前，会提前把样本按照特征大小排好序，默认都放在右子树，然后递归的从小到大拿出一个个的样本放到左子树，然后计算对基于此时的分割点的增益的大小，然后记录并更新最大的增益分割点。

