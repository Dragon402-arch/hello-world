### Support Vector Machine

[教程](https://www.adeveloperdiary.com/data-science/machine-learning/support-vector-machines-for-beginners-linear-svm/)

### 1. 核心概念

#### 1.1 超平面

超平面（Hyperplane）：In a D-dimensional space, a hyperplane is a flat `affine subspace` of dimension (D-1)

在 D 维空间中，超平面是一个 $D-1$ 维的仿射子空间。

在 d 维空间中以向量形式表示的超平面：
$$
\large \begin{align} h(x) =& \beta_1x_1+…+\beta_dx_d+b \\ = & \bigg( \sum_{i=1}^d \beta_ix_i \bigg) +b \\ = & \beta^Tx + b \end{align}
$$
如果 $x_1$ 和 $x_2$ 是超平面上的任意两个点，则有：
$$
\begin{align} h(x_1) = & \beta^T x_1 + b = 0 \\ h(x_2) = & \beta^T x_2 + b = 0 \\ \text{Hence, } \beta^T x_1 + b = & \beta^T x_2 + b \\ \beta^T (x_1 – x_2 ) = & 0 \end{align}
$$
可知向量 $(x_1-x_2)$ 与 $\beta$ 正交， 而向量 $(x_1-x_2)$ 位于超平面上，那么 向量 $\beta$ 是超平面的法向量。

#### 1.2 间隔

间隔（Margin）的定义：

- Margin formally defined as the minimum distance from the decision boundary to the training points.

  间隔通常被定义为从决策边界（超平面）到训练数据样本点的最短距离。

- Margin can be defined using the `minimum distance` (normal distance) from each observations to a given separating hyperplane. 

间隔的作用：使用 margin 来选择最优超平面。

间隔的选择：

- The size of the Margin defines the `confidence` of the classifier, hence the most wide margin is preferable.
- 间隔的大小决定了分类器的信心，因此间隔越大越好。

最大间隔分类器（Maximal Margin Classifier）：最大化从决策边界（超平面）到训练数据样本点的最短距离。

margin类型：

- 函数间隔（Functional Margin）

  对于给定的数据集 D 和超平面 $\beta^Tx+b=0$ ，定义超平面关于样本点 $(x_i,y_i)$ 的函数间隔为
  $$
  margin_f = y_i(\beta^Tx+b)
  $$

- 几何间隔（Geometric Margin ）

  对于给定的数据集 D 和超平面 $\beta^Tx+b=0$ ，定义超平面关于样本点 $(x_i,y_i)$ 的几何间隔为：
  $$
  margin_g = \frac{y_i(\beta^Tx+b)}{|\beta|}
  $$

##### 1.2.1 点到超平面的距离公式

 给定超平面 $\beta^Tx_i+b = 0$ 以及超平面外一点 $P_1(X_i,y_i),y_i=1$，将点$P$ 沿着法向量 $\beta $ 的方向映射（投影）到超平面上的一点$P_0$，所求的距离就是 $|P_0P_1|$,则有：
$$
\begin{aligned}   \beta^T(X_i - |P_0P_1|·\frac{\beta}{|\beta|})+b &= 0 \\
\beta^TX_i - |P_0P_1|·\frac{\beta^T\beta}{|\beta|}+b &= 0 \\
\beta^TX_i - |P_0P_1|·|\beta|+b &= 0
\end {aligned}
$$
则有：
$$
|P_0P_1| = \frac{\beta^TX_i +b }{|\beta|}
$$
将 $y_i = 1$ 和 $y_i=-1$ 的情况统一起来（Finally, we can combine both positive and negative training examples in one equation）：
$$
|P_0P_1| =y_i \bigg(\frac{\beta^TX_i +b }{|\beta|}\bigg)
$$
在所有n个样本点中，我们将线性分类器的间隔（margin）定义为一个点到分离超平面的最小距离：
$$
\large \gamma_i^* =\mathop{\min}\limits_{x_i} \bigg\{ \frac{y_i(\beta^Tx_i+b)}{|\beta|}\bigg\}
$$

注意到 $30x_1+40x_2 – 50= 0$ 与 $3x_1+4x_2 – 5= 0$ 虽然表示的是同一个超平面，但是同一个样本点到超平面的函数间隔却是不同的，而使用上述距离公式计算时距离（几何间隔）却是相同的，如点$(3,4)$ 到两个超平面的距离都是4。

> All the points that achieve this minimum distance are called **Support Vectors** $(x^∗,y^∗)$ for the Hyperplane. Thats the reason this algorithm is named as `Support Vector Machines`.

##### 1.2.2 标准超平面

标准超平面（Canonical Hyperplane，规范超平面）: 如果一个超平面关于支持向量 $x_i^*$ 满足如下条件则称其为标准超平面：
$$
\begin{align} |\beta^T x^*+ b |=1 \end{align}
$$
标准超平面上的一点到决策边界超平面 $\beta^Tx +b = 0  $ 的距离为：
$$
\begin{align} \gamma_i^* = \frac{y^*h(x^*))}{||\beta||} = \frac{y_i^* ( \beta^Tx^*+b)}{||\beta||} = \frac{1}{||\beta||} \end{align}
$$
For each support vector $x_i^*$ we have $y^∗_ih(x^∗_i)=1$ and for any other points which is not a support vector we can define a single combined equation (for both support vectors and other points)：
$$
\begin{align} y_i ( \beta^Tx_i+b) \geq 1 \end{align}
$$
也就是对于支持向量数据点有 $y_i ( \beta^Tx_i+b) = 1 $，对于非支持向量数据点则有 $y_i ( \beta^Tx_i+b) > 1$。

标准超平面实际上就是间隔（margin）确定的超平面。

#### 1.3 其他概念

- 线性可分（linearly separable）

- 法向距离（normal distance）

- 松弛变量（Slack Variable）：The Slack Variable indicates how much the point can violate the margin. In case of `Canonical Hyperplane`, the point may not be at least $\frac{1}{|\beta|}$ away from the hyperplane.

  

到目前为止，假设数据集完全线性可分，而这在实际情况下并不会真正发生。因此，让我们看一下更复杂的情况。我们仍在处理线性SVM，但是，这次类别的某些类别的样本重叠在一起以至于不可能进行完美的线性分离。

线性函数是指：函数中的未知数指数为1，如 z = 3x + 5y + 8

引入松弛变量：
$$
y_i(\beta ^Tx_i+b)\ge1−ξ_i
$$
其中 $ξ_i \ge 0$，是样本 $x_i$ 的松弛变量。

松弛变量帮助定义了三种类型的样本点：

1. 如果 $ξ = 0$，那么样本对应的数据点在 margin上（也就是支持向量）或者离得更远。
2. 如果 $0<ξ<1$，那么样本点 $x_i$ 在margin内部且被正确分类，出现在超平面正确的一侧。$x_i$ is in between the margin and Hyperplane.。
3. 如果 $ξ \ge 1$，那么样本点 $x_i$ 被错误分类了，出现在了超平面错误的一侧。

![SVM](D:\Typora\Notes\NLP\经典模型\SVM.png)

Objective Function needs to also minimize the slack variable (misclassification penalty). 
$$
\large \mathop{\min}\limits_{\beta,b,ξ_i} \bigg\{\frac{|\beta|^2}{2} +C\sum_i^n(ξ_i)^k\bigg\} \\
s.t. \ y_i(\beta ^Tx_i+b)\ge 1-ξ_i,where\ ξ_i \ge 0
$$
`k` is typically set to 1 or 2. If k = 1 , then the loss is named as `Hinge Loss` and if k =2 then its called `Quadratic Loss`.

当 $y_i(\beta ^Tx_i+b) >1$ 时，the classifier predicts the sign correctly (both of them have same sign) and $x_i$ is far from the margin, hence there is no penalty/loss. （分类器预测正确，且 $x_i$ 远在margin之外，因此没有惩罚或损失。此时可以将不等式限制改为等式限制）：

不等式：
$$
y_i(\beta ^Tx_i+b)\ge 1-ξ_i  \\
ξ_i\ge 1-y_i(\beta ^Tx_i+b)
$$
不等式拆分：
$$
ξ_i =
\begin{cases}
1-y_i(\beta ^Tx_i+b), & if\ y_i(\beta ^Tx_i+b)\le1 \\\\\
0, & other
\end{cases}
$$
样本点三种情况：

1. 如果 $y_i(\beta ^Tx_i+b) > 1$ ，则表示分类器预测正确，且 样本点在间隔（margin）之外 ，因此没有惩罚或损失。
2. 如果 $0 \le y_i(\beta ^Tx_i+b) \le 1$，则表示分类器预测正确，但是由于样本点位于间隔和超平面之间，仍然需要一点惩罚，且越靠近超平面惩罚越大。
3. 如果 $y_i(\beta ^Tx_i+b) < 0$，则表示分类器预测错误，惩罚会比较大，且样本点在错误一侧越远离决策边界超平面，损失和惩罚也会线性增长。

根据等式：以上分析，可得：
$$
ξ_i=max\bigg(0,1-y_i(\beta ^Tx_i+b)\bigg)
$$
目标函数变为：
$$
L = \frac{|\beta|^2}{2} +C\sum_{i=1}^nmax\bigg(0,1-y_i(\beta ^Tx_i+b)\bigg)
$$





### 2. 原问题与对偶问题

原问题和对偶问题（Primal Problem and Dual Problem）

- 弱对偶性（weak duality ）：$p^∗−d^∗>0$
- 强对偶性（strong duality）：$p^*-d^*=0$
- KKT条件（KKT condition）：使得强对偶性成立的条件
- 拉格朗日乘数（ Lagrange Multiplier）


$$
\begin{align}
\min_x f(x) & \\
\text{s.t. } g_i(x) \leq 0 & \text{      ,       } \forall i = 1..n \\
h_i(x) = 0 & \text{      ,        } \forall j = 1..m \\  
\end{align}
$$
拉格朗日函数可写为：
$$
\begin{align}
L(x,\alpha,\lambda ) = f(x) +  \sum_{i=1}^n \alpha_i g_i(x)+ \sum_{j=1}^m \lambda_j h_j(x) \\
\end{align}
$$
#### 2.1 KKT条件

**The KKT conditions are composed of：**

1.  Primal feasibility (inequalities) 原问题可行性（不等式）
   $$
   g_i(x) \leq 0  \text{      ,       } \forall i = 1..n
   $$

2. Primal feasibility (equalities) 原问题可行性（等式）
   $$
   \frac{\part L(x,\alpha,\lambda )}{\part \lambda_i} = h_i(x) = 0  \text{      ,        } \forall j = 1..m
   $$

3. Dual feasibility 对偶问题可行性
   $$
   \alpha_i \geq  0 \text{    ,        } \forall i = 1..n
   $$

4. Complementary Slackness 互补松弛条件
   $$
   \alpha_i g_i(x) = 0  \text{    ,        } \forall i = 1..n
   $$

5.  Stationarity 稳定性条件
   $$
   \frac{\part L(x,\alpha,\lambda )}{\part x} = 0  \text{    ,        } \forall d = 1..D
   $$

**以上5个条件被称为KKT条件，在强对偶问题中必须都满足。**

原问题（primal problem）：
$$
p^* = \min \limits_{x} \max \limits_{\alpha_i\ge0,\lambda} L(x,\alpha,\lambda )  = \min \limits_{x}q(x)
$$
对偶问题 （dual problem）：
$$
d^* = \max \limits_{\alpha_i\ge0,\lambda}\min \limits_{x} L(x,\alpha,\lambda )  = \max \limits_{\alpha_i\ge0,\lambda}g(\alpha, \lambda)
$$

### 3.线性支持向量机

#### 3.1 硬间隔分类器

硬间隔分类器（Hard Margin Classifier）假定训练样本是完全线性可分的，也就是所有样本都要划分正确，对离群值比较敏感。

原始问题的标准形式为：
$$
\begin{align}
\text{Objective Function : } \min_{\beta,b} \Big \{ \frac{||\beta^2||}{2} \Big \} \\
\text{s.t Linear Constraint : } 1- y_i ( \beta^Tx_i+b) \le  0  \text{   ,   } \forall x_i \in D
\end{align}
$$

原始问题变量和对偶问题变量划分：

- 原始问题变量： $\beta,b$
- 对偶问题变量
  - $\alpha_i$ 对应约束条件： $ 1 - y_i ( \beta^Tx_i+b) \le 0 $

对原始问题的目标函数使用拉格朗日乘数法可得到其对偶问题，具体来讲，也就是对每个约束条件添加拉格朗日乘数 $\alpha_i$ ，则该问题的拉格朗日函数可写为：
$$
\begin{align}
L(\beta,b,\alpha) & = \frac{||\beta^2||}{2} +  \sum_{i=1}^n \alpha_i (1- y_i(\beta^Tx_i+b)) \\
& = \frac{||\beta^2||}{2} –  \sum_{i=1}^n \alpha_i (y_i(\beta^Tx_i+b)-1) \\
\end{align}
$$
因此，其对偶问题为：
$$
\large \max_{\alpha\ge0} g(\alpha)
$$
其中 
$$
g(\alpha) =\large \min_{\beta,b} \frac{||\beta^2||}{2} - \sum_{i=1}^n \alpha_i (y_i(\beta^Tx_i+b)) +  \sum_{i=1}^n \alpha_i
$$
求偏导并令其为零：
$$
\frac{\part L}{\part \beta}= \frac{\part L}{\part b}  = 0
$$
可得：
$$
\begin{align}
\frac{\part L}{\part \beta} & = \beta – \sum_{i=1}^n \alpha_i y_i x_i = 0 \\
\beta & = \sum_{i=1}^n \alpha_i y_i x_i \\
 \frac{\part L}{\part b}  & = \sum_{i=1}^n \alpha_i y_i = 0 
\end{align}
$$
将上面的条件带入拉格朗日函数中可得：
$$
\begin{align}
 g(\alpha) & = \min_{\beta,b} L(\beta,b,\alpha)  \\ 
=& \frac{||\beta^2||}{2} –  \sum_{i=1}^n \alpha_i (y_i(\beta^Tx_i+b)-1) \\
=& \frac{1}{2} \beta^T\beta –  \sum_{i=1}^n \alpha_i y_i\beta^Tx_i – \sum_{i=1}^n \alpha_i y_ib + \sum_{i=1}^n \alpha_i \\
=& \frac{1}{2} \beta^T\beta –  \beta^T \Big ( \sum_{i=1}^n \alpha_i y_ix_i \Big ) – b \Big ( \sum_{i=1}^n \alpha_i y_i \Big ) + \sum_{i=1}^n \alpha_i \\
=& \frac{1}{2} \beta^T\beta –  \beta^T \Big ( \beta \Big ) – b \Big ( 0 \Big ) + \sum_{i=1}^n \alpha_i \\
=& – \frac{1}{2} \beta^T\beta + \sum_{i=1}^n \alpha_i \\
=& \sum_{i=1}^n \alpha_i – \frac{1}{2} \sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_iy_jx_i^Tx_j
\end{align}
$$
对偶问题目标函数的最终形式可以写为：
$$
\begin{align}
\textbf{Objective Function: } & \max_{\alpha} g(\alpha) = \sum_{i=1}^n \alpha_i – \frac{1}{2} \sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_iy_jx_i^Tx_j \\
\textbf{Linear Constraints: } & \alpha_i \geq 0, \forall i \in D , \text{ and } \sum_{i=1}^n \alpha_i y_i = 0
\end{align}
$$
从中可以求解得到 $\alpha_i$，使用SMO算法求解。

Given optimal primal and dual values, the following KKT conditions are enforced:
$$
\begin{align}
\alpha_i (1- y_i(\beta^Tx_i+b)) = 0 \\
\text{and  } \alpha_i \geq 0
\end{align}
$$
若 $\alpha_i \not=  0$ ，则有 $y_i(\beta^Tx_i+b)=1$，所对应的样本点必定位于最大间隔边界上，是一个支持向量。

若 $\alpha_i = 0$ ，则有 $y_i(\beta^Tx_i+b)\ge 1$，样本 $x_i$ 不会在确定参数$\beta$ 时发挥作用，也就不会对确定 $\beta^Tx+b$有任何影响。

一旦知道了每个数据点的 $\alpha_i$ ，进行如下就和（本质上是对所有支持向量数据点求和）就可求得 $\beta$：
$$
\begin{align}
\large \beta = \sum_{i , \alpha_i \geq 0} \alpha_i y_ix_i
\end{align}
$$
注意到对于任意支持向量数据点都有：$y_i(\beta^Tx_i+b)=1$，理论上，可以任选一个支持向量数据点带入该式子求和得到 $b$ ，但是现实任务中采用了一种更具有鲁棒性的做法：使用所有支持向量数据点求解得到 $b$ 的平均值：
$$
b = \frac{1}{|S|}\sum_{x_i \in S}(\frac{1}{y_i} - \beta^Tx_i)
$$
由于 $y_i \in \{-1,1 \}$，则有 $y_i = \frac{1}{\large y_i} $，于是上式可以转换为：
$$
b = \frac{1}{|S|}\sum_{x_i \in S}(y_i- \beta^Tx_i)
$$

#### 3.2 软间隔分类器

软间隔分类器（Soft Margin Classifie）假定训练样本是线性可分的，但也允许一些样本划分错误，并要求划分错误的样本尽可能少，也就是近似线性可分。

软间隔分类器基础：

![软间隔分类器](D:\Typora\Notes\NLP\经典模型\软间隔分类器.png)

原始问题的标准形式为：
$$
\begin{align}
\text{Objective Function : } \min_{\beta,b,\xi} \Big \{ \frac{||\beta^2||}{2} + C \sum_{i=1}^n (\xi_i)^k \Big \} \\
\ s.t.\  1 – \xi_i - y_i ( \beta^Tx_i+b) \le 0  \\
-\xi_i \leq 0
\end{align}
$$
原始问题变量和对偶问题变量划分：

- 原始问题变量： $\beta,b,\xi_i$
- 对偶问题变量
  - $\alpha_i$ 对应约束条件 $ 1 – \xi_i - y_i ( \beta^Tx_i+b) \le 0 $
  - $\lambda_i$  对应约束条件 $-\xi_i \leq 0$

原始问题的拉格朗日函数为：
$$
\begin{align}
L(\beta,b,\xi,\alpha,\lambda) &=  \frac{||\beta^2||}{2} + C \sum_{i=1}^n \xi_i + \sum_{i=1}^n \alpha_i (1 – \xi_i- y_i(\beta^Tx_i+b)) + \sum_{i=1}^n \lambda_i (-\xi_i) \\
&= \frac{||\beta^2||}{2} - \sum_{i=1}^n \alpha_i (y_i(\beta^Tx_i+b)) +  \sum_{i=1}^n \alpha_i +  \sum_{i=1}^n (C-\alpha_i-\lambda_i)\xi_i
\end{align}
$$
因此，其对偶问题为：
$$
\large \max_{\alpha\ge0,\lambda\ge0} g(\alpha,\lambda)
$$
其中 
$$
g(\alpha,\lambda) =\large \min_{\beta,b,\xi} \frac{||\beta^2||}{2} - \sum_{i=1}^n \alpha_i (y_i(\beta^Tx_i+b)) +  \sum_{i=1}^n \alpha_i +  \sum_{i=1}^n (C-\alpha_i-\lambda_i)\xi_i
$$
Let’s use the KKT conditions to find the optimal dual variables.
$$
\frac{\part L}{\part \beta}= \frac{\part L}{\part b} = \frac{\part L}{\part \xi_i} = 0
$$
注意这里是因为 $ C \sum_{i=1}^n \xi_i$ 出现在了拉格朗日函数中所以需要对其进行求导，同时 $\xi_i\le 0$ 又作为不等式条件存在 。

可得：
$$
\begin{align}
\frac{\part L}{\part \beta} & = \beta – \sum_{i=1}^n \alpha_i y_i x_i = 0 \\
\beta & = \sum_{i=1}^n \alpha_i y_i x_i \\
 \frac{\part L}{\part b}  & = \sum_{i=1}^n \alpha_i y_i = 0 \\
 \frac{\part L}{\part \xi_i}  & = C- \alpha_i-\lambda_i = 0
\end{align}
$$
将上面的条件带入拉格朗日函数中可得：
$$
\begin{align}
g(\alpha,\lambda) & = \min_{\beta,b,\xi} L(\beta,b,\xi,\alpha,\lambda)  \\ 
&= \frac{||\beta^2||}{2} + C \sum_{i=1}^n \xi_i – \sum_{i=1}^n \alpha_i (y_i(\beta^Tx_i+b) – 1 + \xi_i ) – \sum_{i=1}^n \lambda_i \xi_i \\
&= \frac{||\beta^2||}{2} - \sum_{i=1}^n \alpha_i (y_i(\beta^Tx_i+b)) +  \sum_{i=1}^n \alpha_i +  \sum_{i=1}^n (C-\alpha_i-\lambda_i)\xi_i \\
&= \frac{1}{2} \beta^T\beta -\beta^T \bigg (  \sum_{i=1}^n \alpha_i y_i x_i \bigg ) -b \sum_{i=1}^n \alpha_i y_i +\sum_{i=1}^n \alpha_i   + \sum_{i=1}^n \bigg ( C – \alpha_i – \lambda_i  \bigg) \xi_i \\
&= \frac{1}{2} \beta^T\beta -\beta^T \bigg (  \beta \bigg ) -b \bigg( 0 \bigg ) +\sum_{i=1}^n \alpha_i   + \sum_{i=1}^n \bigg ( 0 \bigg) \xi_i \\
&= \sum_{i=1}^n \alpha_i  – \frac{1}{2} \beta^T\beta \\
&= \sum_{i=1}^n \alpha_i  – \frac{1}{2} \sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_iy_jx_i^Tx_j \\
\end{align}
$$
对偶问题函数的最终形式可以写为：
$$
\begin{align}
\textbf{Objective Function: } & \max_{\alpha} g(\alpha,\lambda) = \sum_{i=1}^n \alpha_i – \frac{1}{2} \sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_iy_jx_i^Tx_j \\
\textbf{Linear Constraints: } & 0 \leq \alpha_i \leq C, \forall i \in D , \text{ and } \sum_{i=1}^n \alpha_i y_i = 0
\end{align}
$$
使用SMO算法求解 $\alpha_i$。
给定最优原始值和最优对偶值，满足以下KKT条件：

- Stationarity 稳定性条件
  $$
  C- \alpha_i-\lambda_i = 0
  $$

- Complementary slackness 互补松弛条件

  不等式 $y_i ( \beta^Tx_i+b) \geq  1 – \xi_i$ 产生的互补松弛条件为：
  $$
  \alpha_i (1 – \xi_i- y_i(\beta^Tx_i+b)) = 0 \\
  $$
  不等式 $\xi_i \geq 0$ 产生的互补松弛条件为：
  $$
  \begin{align}
  \lambda_i (0-\xi_i)= \lambda_i \xi_i = 0 \\
  \end{align}
  $$

1. 当 $\alpha_i=0$ 时，则 $\lambda_i= C>0$，进一步则有 $\xi_i = 0$ 
    - $y_i(\beta^Tx_i+b)\ge 1$ ，on or outside the margin
         - $y_i(\beta^Tx_i+b)= 1$ ，on  the margin
         - $y_i(\beta^Tx_i+b)> 1$ ，outside the margin
2. 当 $\alpha_i=C$ 时，则  $\lambda_i= 0$，此时 $\xi_i$ 无法确定，情况如下：
    - $y_i(\beta^Tx_i+b)= 1$ ，on  the margin
    - $y_i(\beta^Tx_i+b)> 1$ ，outside  the margin
    - $y_i(\beta^Tx_i+b)< 1$ ，inside  the margin

3. 当 $ {0< \alpha_i<C}$ 时，则  $\lambda_i \not=  0$，进一步则有 $\xi_i = 0, y_i(\beta^Tx_i+b)=1$ ，此时样本数据点正好在margin上。

根据以上信息，原始问题变量$\beta,b,\xi_i$ 的最优解为：
$$
\begin{align} \beta &= \sum_{i=1}^n \alpha_i y_ix_i \\
 b &=  y_i - \beta^Tx_i, \ if \ 0<\alpha_i < C,\\
\xi_i &=
\begin{cases}
1-y_i( \beta^Tx_i+b), & if\ \alpha_i = C \\
0 & other
\end{cases}

\end {align}
$$

### 4. 非线性支持向量机

#### 4.1 映射函数与核函数

设 X 是 输入空间，H是特征空间，如果存在一个从 X 到 H 的映射：
$$
\phi(x)：X\rightarrow H
$$
使得对于所有的 $x,z \in X$ ，函数$K(x,z)$ 满足条件：
$$
K(x,z) =\phi(x)·\phi(z)
$$
则称 $K(x,z)$  是核函数，$\phi(x)$ 为映射函数。

**一般只会定义一个核函数，而不去定义一个映射函数，因为通常直接计算核函数值是比较容易的，而如果通过计算映射函数值再去计算核函数值，则是比较困难的。**当训练数据样本点在低维样本空间中线性不可分时，此时可以将样本从原始空间映射到一个更高维空间，使得样本在该高维空间线性可分。在映射到高维空间后，由于特征空间维数可能很高，甚至可能是无穷维，直接在高维空间进行计算比较困难，**此时可以使用核函数直接在原始样本空间进行计算，从而达到相同的效果**。

#### 4.2 参数求解过程

核函数（Kernel Function），如线性核函数：
$$
K(x_i,x_j) = < \phi(x_i),\phi(x_j) > =  \phi(x_i)^T \phi(x_j)
$$
对偶问题目标函数为：
$$
\begin{align}
\max_{\alpha} L_{dual} & = \sum_{i=1}^n \alpha_i – \frac{1}{2} \sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_iy_jx_i^Tx_j \\
& = \sum_{i=1}^n \alpha_i – \frac{1}{2} \sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_iy_j \phi(x_i)^T\phi(x_j) \\
& = \sum_{i=1}^n \alpha_i – \frac{1}{2} \sum_{i=1}^n\sum_{j=1}^n \alpha_i \alpha_j y_iy_j K(x_i, x_j) \\
\end{align}
$$
在已知 $\alpha_i$ 的情况下求解 $\beta,b$。

求解 $\beta$
$$
\begin{align}
\beta & = \sum_{i , \alpha_i \geq 0} \alpha_i y_ix_i \\
& = \sum_{i , \alpha_i \geq 0} \alpha_i y_i \phi(x_i) \\
\end{align}
$$
求解 $b$
$$
\begin{align}
b &= \frac{1}{|S|}\sum_{x_i \in S}(y_i- \beta^Tx_i) \\
& =  \frac{1}{|S|}\sum_{x_i \in S}\bigg(y_i – \sum_{j , \alpha_j \geq 0} \alpha_j y_j \phi(x_j)^T \phi(x_i ) \bigg) \\
& =  \frac{1}{|S|}\sum_{x_i \in S}\bigg(y_i – \sum_{j , \alpha_j \geq 0} \alpha_j y_j K(x_j, x_i ) \bigg) \\
\end{align}
$$
做预测：
$$
\begin{align}
\hat{y} & = \text{sign}( \beta^Tz_i +b ) \\
& = \text{sign}( \beta^T \phi(z) +b ) \\
& = sign \bigg ( \sum_{i , \alpha_i \geq 0} \alpha_i y_i \phi(x_i)^T \phi(z_i) + b \bigg ) \\
& = sign \bigg ( \sum_{i , \alpha_i \geq 0} \alpha_i y_i K(x_i , z_i) + b \bigg ) \\
& = sign \bigg ( \sum_{i , \alpha_i \geq 0} \alpha_i y_i K(x_i , z_i) + \frac{1}{|S|}\sum_{x_i \in S}\bigg(y_i – \sum_{j , \alpha_j \geq 0} \alpha_j y_j K(x_j, x_i ) \bigg)  \bigg ) \\


\end{align}
$$

### 5 支持向量回归





