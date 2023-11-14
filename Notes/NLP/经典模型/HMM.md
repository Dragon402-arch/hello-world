# Hidden Markov Model

- [新手入门教程](https://www.adeveloperdiary.com/data-science/machine-learning/introduction-to-hidden-markov-model/)

- [讲解](https://towardsdatascience.com/introduction-to-hidden-markov-models-cd2c93e6b781)
- [讲解2](https://github.com/luopeixiang/named_entity_recognition/blob/master/models/hmm.py)

## 1. 基本概念

- 状态变量（state variable）

  该变量通常是不可被观测的，因此也称为隐变量（hidden variable）。

  **假设条件**：当前时刻的状态仅由前一个时刻的状态决定（一阶马尔科夫性）。

- 观测变量（observe variable）

  **假设条件**：在任一时刻，观测变量的取值只由对应时刻的状态变量决定。

- **状态转移概率矩阵**（Transition Probability Matrix）

  状态变量的取值一般为离散值，假定可取 N 个离散值，则将形成一个 $(N,N)$  的状态转移概率矩阵，示例如下：
  $$
  A =  \begin{bmatrix}a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}
  $$
  其中 $a_{ij}$  定义如下：
  $$
  a_{ij} = P(y_{t+1}=s_j|y_{t}=s_i)
  $$
  状态转移概率矩阵表示的当前与之后两个时刻状态间的转移概率情况，比如当前时刻状态变量取值为 $s_1$ ，而下一时刻状态变量可能 取值为$s_1,s_2,s_3$ 中的一个 ，则必定有 $a_{11}+a_{12}+a_{13} = 1$ 。更一般的情况是：
  $$
  \sum_{j=1}^{M} a_{ij} = 1 \; \; \;   \forall i
  $$

- **初始状态概率分布**（Initial State Probability Distribution）：

  初始时刻状态变量取值的概率分布，通常记为 $\pi = (\pi_1,\pi_2,…,\pi_N)$，其中：
  $$
  \pi_i = P(y_1 = s_1)
  $$
  由概率的性质可知：
  $$
  \sum_{i=1}^N \pi_i = 1
  $$

- **发射概率矩阵**（Emission Probability Matrix）

  根据当前时刻的状态变量取值获得各个观测值的概率，假定状态变量有 N 个离散取值，观测变量有 M 个离散取值，则将形成一个$(N,M)$ 的发射概率矩阵，示例如下：
  $$
  B = \begin{bmatrix} 
  b_{11} &  b_{12} \\ 
  b_{21} &  b_{22} \\ 
  b_{31} &  b_{32} 
  \end{bmatrix}
  $$
  其中 $b_{ij}$ 定义如下：
  $$
  b_{ij} = P(x_t=o_j|y_t=s_i)
  $$
  发射概率矩阵表示的是在状态变量取值给定的情况下，产生某个观测值的概率情况，比如当前时刻状态变量取值为 $s_1$ ，而当前时刻观测变量的可能取值为 $o_1,o_2$ 中的一个 ，则必定有 $b_{11}+b_{12} = 1$ 。更一般的情况是：
  $$
  \sum_{j=1}^M b_{ij} = 1
  $$

## 2. 三大经典问题

### 2.1 概率计算问题：Evaluation Problem

问题描述：给定模型 $\lambda = (A,B,\pi)$ 以及观测序列 $O=(o_1,o_2,…,o_T)$ ，求解模型生成该观测序列的概率值 $P(O|\lambda)$。

对应算法：前向算法（Forward Algorithm）或后向算法（Backward Algorithm），是一个动态规划算法。

直接计算：

1. First we need to find all possible sequences of the state 
2. Then from all those sequences , find the probability of which sequence generated the observe sequence.
3. Mathematically,$P(O|\lambda)$ can be estimated as $P(O|\lambda)= \sum_{S\in S^T}P(O,S|\lambda)=\sum_{S\in S^T}P(O|\lambda,S)P(S)$ 

给定序列长度为 $T$ 以及状态变量取值个数为 $N$ ，则可能存在 $T^N$ 个序列，而在实际任务中，$T,N$ 都是比较大的值，算法复杂度为$O(TN^T)$，因此不能使用上面的方式计算概率，可以使用一种复杂度为 $O(N^2T)$ 的前向算法或后向算法来计算，该算法是一种动态规划算法。

####  2.1.1 前向算法（Forward Algorithm）：

定义：
$$
\alpha_t(i) = P(o_1,o_2,…,o_t,s_t=i|\lambda)
$$
其中 $s_t =j $ 表示状态序列的第 $t$ 个状态的取值为状态值 $j$ 。

则有：

1. Initialization：
   $$
   \alpha_1(i) = \pi_i b_i(o_1)  \text{ ,  }1\le i \le N
   $$

2. Recursion：
   $$
   \alpha_{t+1}(j) = \sum_{i=1}^N \alpha_{t}(i)a_{ij}b_j(o_{t+1}) \text{ ,  }1\le j \le N\text{ ,  }1\le t \le T\
   $$

3. Termination：
   $$
   P(O|\lambda) = \sum_{i=1}^N \alpha_T(i)
   $$

推导过程：

首先是 $\alpha_1(i)$ :
$$
\alpha_1(i) = P(o_1,s_1=i|\lambda) =  P(o_1|\lambda,s_1=i) P(s_1=i|\lambda) =b_i(o_1)\pi_i
$$
其次是 $\alpha_{t+1}(j)$:
$$
\begin{align} \alpha_{t+1}(j) &= P(o_1,o_2,…,o_{t+1},s_{t+1}=j|\lambda) \\
 &= \sum_{i=1}^NP(o_1,o_2,…,o_{t+1},s_{t}=i,s_{t+1}=j|\lambda) \\
 &= \sum_{i=1}^NP(o_{t+1}|o_1,o_2,…,o_{t},s_{t}=i,s_{t+1}=j,\lambda)P(o_1,o_2,…,o_{t},s_{t}=i,s_{t+1}=j,\lambda) \\
  &= \sum_{i=1}^NP(o_{t+1}|s_{t+1}=j,\lambda)P(o_1,o_2,…,o_{t},s_{t}=i,s_{t+1}=j,\lambda) \\
  &= \sum_{i=1}^NP(o_{t+1}|s_{t+1}=j,\lambda)P(s_{t+1}=j|o_1,o_2,…,o_{t},s_{t}=i,\lambda)P(o_1,o_2,…,o_{t},s_{t}=i|\lambda) \\
   &= \sum_{i=1}^N b_j({o_{t+1}})P(s_{t+1}=j|s_{t}=i,\lambda)\alpha_t(i) \\
    &= \sum_{i=1}^N b_j({o_{t+1}})a_{ij}\alpha_t(i) \\
\end {align}
$$
​	注解 ：$P(A,B)=P(A|B)P(B)$，此外 $b_j(o_{t+1})$ 与求和项无关，可以提到前面。

最后是 $P(O|\lambda)$：
$$
P(O|\lambda) = \sum_{i=1}^N P(o_1,…,o_T,s_t=i|\lambda) = \sum_{i=1}^N \alpha_T(i)
$$

#### 2.1.2 后向算法（Backward Algorithm）

定义：
$$
\beta_t(i) = P(o_{t+1},o_{t+2},…,o_T|s_t=i,\lambda)
$$
则有：

1. Initialization：
   $$
   \beta_T(i) = 1  \text{ ,  }1\le i \le N
   $$

2. Recursion：
   $$
   \beta_{t}(i) = \sum_{j=1}^N \beta_{t+1}(j)a_{ij}b_j(o_{t+1}) \text{ ,  }1\le j \le N\text{ ,  }1\le t \le T\
   $$

3. Termination：
   $$
   P(O|\lambda) = \sum_{i=1}^N \beta_1(i)\pi_ib_i(o_{1})
   $$

推导过程：

推导过程：

首先是 $\beta_T(i)$ :
$$
\beta_T(i) = \sum_{m=1}^M P(o_T=m|s_T=i,\lambda) =\sum_{m=1}^M b_im=1
$$
其次是 $\beta_{t}(i)$:
$$
\begin{align} \beta_{t}(i) &=  P(o_{t+1},o_{t+2},…,o_T|s_t=i,\lambda)\\
 &= \sum_{j=1}^NP(o_{t+1},o_{t+2},…,o_T,s_{t+1}=j|s_{t}=i,\lambda) \\
  &= \sum_{j=1}^N P(o_{t+1},o_{t+2},…,o_T|s_{t+1}=j,s_{t}=i,\lambda)P(s_{t+1}=j|s_{t}=i,\lambda) \\
   &= \sum_{j=1}^N P(o_{t+1},o_{t+2},…,o_T|s_{t+1}=j,\lambda)P(s_{t+1}=j|s_{t}=i,\lambda) \\
    &= \sum_{j=1}^N P(o_{t+1}|s_{t+1}=j,\lambda)P(o_{t+2},…,o_T|s_{t+1}=j,\lambda)P(s_{t+1}=j|s_{t}=i,\lambda) \\
&= \sum_{j=1}^N  b_j(o_{t+1})   \beta_{t+1}(j)a_{ij}
\end {align}
$$
最后是 $P(O|\lambda)$：
$$
\begin{align} P(O|\lambda) &=\sum_{i=1}^N P(o_{1},o_{2},o_{3},…,o_T,s_1=i|\lambda)\\
&=\sum_{i=1}^N P(o_{1},o_{2},o_{3},…,o_T|s_1=i,\lambda)P(s_1=i|\lambda) \\
&=\sum_{i=1}^N P(o_{1}|s_1=i,\lambda)P(o_{2},o_{3},…,o_T|s_1=i,\lambda)P(s_1=i|\lambda) \\
&=\sum_{i=1}^N b_i(o_1)\beta_1(i)\pi_i  \\
\end {align}
$$

### 2.2 参数学习问题：Learning Problem

[监督学习推导教程](https://www.adeveloperdiary.com/data-science/machine-learning/derivation-and-implementation-of-baum-welch-algorithm-for-hidden-markov-model/)

HMM的学习算法根据训练数据的不同，可以分为有监督学习和无监督学习两种。

如果训练数据中同时包含观测序列和状态序列，则对应有监督学习算法，而如果是只包含观测数据，则对应无监督学习算法。

**有监督学习算法参数学习**

问题描述：令  $\lambda = (A,B,\pi)$，给定观测序列 $O=(o_1,o_2,…,o_T)$ ，求解 $\lambda^*= argmax_{\lambda} P(O|\lambda)$

对应算法：Baum-Welch Algorithm or Forward-Backward Algorithm（前向后向算法），是EM算法的一种特殊情况。

Estimate for $a_{ij},b_{jk}$ ：
$$
{\hat a_{ij}} = \frac{\text{expected number of transitions from hidden state i to state j}}{\text{expected number of transition from hidden state i}} \\

{\hat b_{jk}} = \frac{\text{expected number of times in hidden state j and observing o(k) }}{\text{expected number of times in hidden state j}}
$$

#### 2.2.1 求解 $\hat a_{ij}$

If we know the probability of a given transition from `i` to `j` at time step `t`, then we can sum over all the `T` times to estimate for the numerator in our equation for $\hat A$.

首先定义$\xi_t(ij)$：
$$
\xi_t(i,j) = P(s_t=i,s_{t+1}=j|O,\lambda)
$$
于是
$$
\begin{align} 
\large \hat a_{ij} &=  \frac{\sum_{t=1}^{T-1}P(s_t=i,s_{t+1}=j|O,\lambda)}{\sum_{t=1}^{T-1}\sum_{k=1}^NP(s_t=i,s_{t+1}=k|O,\lambda)} \\
&=  \frac{\sum_{t=1}^{T-1}\xi_t(i,j)}{\sum_{t=1}^{T-1}\sum_{k=1}^N\xi_t(i,k)} \\
\end {align}
$$
其中
$$
\xi_t(i,j) = P(s_t=i,s_{t+1}=j|O,\lambda) = \frac{ P(s_t=i,s_{t+1}=j,O|\lambda) }{ P(O|\lambda) } = \frac{ P(s_t=i,s_{t+1}=j,O|\lambda) }{\sum_{i=1}^N\sum_{j=1}^N P(s_t=i,s_{t+1}=j,O|\lambda)  }
$$ { }
而 
$$
\begin{align}  P(s_t=i,s_{t+1}=j,O|\lambda) &= P(o_1,…,o_t,s_t=i)P(o_{t+1},…,o_T,s_{t+1}=j|o_1,…,o_t,s_t=i,\lambda) \\
&= \alpha_t(i)P(o_{t+1},…,o_T,s_{t+1}=j|s_t=i,\lambda) \\
&= \alpha_t(i)P(s_{t+1}=j|s_t=i,\lambda)P(o_{t+1},…,o_T|s_{t+1}=j,s_t=i,\lambda)\\
&= \alpha_t(i)a_{ij}P(o_{t+1},…,o_T|s_{t+1}=j,\lambda)\\
&= \alpha_t(i)a_{ij}\beta_{t+1}(j)\\
\end {align}
$$

#### 2.2.2 求解 $\hat b_{jk}$

首先定义 $\gamma_t(i)$
$$
\gamma_t(i) = P(s_t=i|O,\lambda)
$$
于是：
$$
\hat{b_{jk}} = \frac{\sum_{t=1}^T \gamma_t(j) I(o_t=k)}{\sum_{t=1}^T \gamma_t(j) }
$$
其中 $I(o_t=k)$ 是指示函数，若 $o_t=k$ 则其取值为1，否则取值为 0。

其中
$$
\gamma_t(i) = P(s_t=i|O,\lambda) = \frac{ P(s_t=i,O|\lambda) }{ P(O|\lambda) } = \frac{ P(s_t=i,O|\lambda) }{\sum_{i=1}^N P(s_t=i,O|\lambda)  }
$$
而
$$
\begin{align}  P(s_t=i,O|\lambda) &= P(o_1,…,o_t,s_t=i)P(o_{t+1},…,o_T|o_1,…,o_t,s_t=i,\lambda) \\
&= \alpha_t(i) P(o_{t+1},…,o_T|s_t=i,\lambda) \\
&= \alpha_t(i) \beta_t(i)
\end {align}
$$

#### 2.2.3 EM算法应用

1. 初始化参数 A，B

2. 迭代直至收敛

   - E-step：计算 $\xi_t(i,j),\gamma_t(i)$

   - M-step：计算 $ \hat a_{ij},\hat b_{j,k} $

3. 返回参数 A 和 B

#### 2.2.4 无监督学习算法

由于训练数据中只包含观测序列数据，而没有状态序列数据，因此是不能直接使用频率来估计概率的。此时会有专门的 Baum-Welch 算法针对这种情况来求解模型参数。

### 2.3  解码问题：Decoding Problem

问题描述：给定HMM模型参数  $\lambda = (A,B,\pi)$ 以及观测序列 $O=(o_1,o_2,…,o_T)$ ，求解最大概率的状态序列 $  S = (s_1,s_2,…,s_T)$。

对应算法：维特比算法（Viterbi Algorithm）

现实应用：分词、词性标注、实体识别、序列标注

与概率计算问题类似，可以首先获取所有可能的隐状态序列，然后分别计算每个序列对应的概率，由此找到最大概率的隐状态序列。但是该方法的算法复杂度较高 $$O(N^T·T)$$，计算效率较低，此处可以使用维特比算法进行求解。

定义: 已知观测序列 $$o_1 ,o_2 ...o_t$$ 以及 t 时刻的状态为 $j$，计算出现该情况的最大概率，该概率可以定义如下：
$$
w_t (j) = max_{s_1,s_2,...,s_{t−1}}P(s_1 ...,s_{t−1} ,s_t = j|λ)
$$
假设我们现在已经知道了观测序列 $$o_1 ,o_2 ...o_{t-1}$$ 以及 $t-1$ 时刻的状态为 $i$ 情况的概率为 $$ w_{t−1}$$，则观测序列 $$o_1 ,o_2 ...o_t$$ 以及 t 时刻的状态为 $j$ 情况的概率为 $$max_{1≤i≤N} w_{t−1} (i) a_{ij} b_j (o_t )$$，于是可得如下递推式：
$$
w_t(j) = max_{1≤i≤N} w_{t−1} (i) a_{ij} b_j (o_t )
$$
关于带上$$b_j (o_t )$$的原因是因为观测值$$o_t$$是已知的，t 时刻状态的取值不同，发射观测值$$o_t$$的概率值也是不同的，因此需要在获得了 t 时刻状态概率的基础上再乘上一个发射概率，才是符合定义情况的真实概率。

需要一个N * T 的矩阵用于存储每个时刻的值，在 t = 1 时，不存在转移，因此使用初始状态分布进行填充，当 $t >1$ 时，$w_t(i)$ 的值为取$t-1$ 时刻的N个状态在 $t$ 时刻转移到状态 $i$ 的概率中最大的概率，同时使用另一个矩阵保存前一个时刻的状态索引，以便于进行回溯。

## 3. EM 算法推导

EM（Expectation Maximization）算法整体公式为：
$$
θ^{(t+1)} = \mathop {argmax}_{θ} \int_Z\log(P(X,Z|θ)P(Z|X,θ^{(t)}))dZ\
$$
其中 $Z$ 是一个隐变量。

### 3.1 算法步骤

- Expectation-step 
  $$
  Q(θ,θ^{(t)}) =E_{Z|X,θ^{(t)}} \bigg[log(P(X,Z|θ) \bigg] = \int_Z\ P(Z|X,θ^{(t)}log(P(X,Z|θ)))dZ
  $$

- Maximization step
  $$
  θ^{(t+1)} = \mathop {argmax}_{θ}\ Q(θ,θ^{(t)}) =\mathop {argmax}_{θ} \int_Z\log(P(X,Z|θ)P(Z|X,θ^{(t)}))dZ
  $$
  

### 3.2 公式证明
由 $P(X,Z)=P(X)P(Z|X)$ 可得：
$$
P(X) = \frac{P(X,Z)}{P(Z|X)}
$$
上式两边取对数，则有：
$$
logP(X|θ) = logP(X,Z|θ) - logP(Z|X,θ)
$$
再对上式两边对随机变量$$Z$$取期望，左边不包含变量$$Z$$，因此不变，等式变为：
$$
\begin{aligned}logP(X|θ) &= \int_Z\log(P(X,Z|θ)P(Z|X,θ^{(t)}))dZ - \int_Z\log(P(Z|X,θ)P(Z|X,θ^{(t)}))dZ \\ &= Q(θ,θ^{(t)}) - H(θ,θ^{(t))})\end{aligned}
$$
优化的目标是最大化 $P(X|\theta)$  ，因此希望 $\theta$ 在每次迭代后 $P(X|\theta)$ 的值可以增加
$$
logP(X|θ^{(t+1)}) -logP(X|θ^{(t)})=Q(θ^{(t+1)},θ^{(t)})- Q(θ^{(t)},θ^{(t)})+ H(θ^{(t)},θ^{(t))})-H(θ^{(t+1)},θ^{(t))})
$$
我们通过最大化 $ Q(θ,θ^{(t)})$ 从而得到一个 $θ^{(t+1)}$，同时希望有：
$$
P(X|θ^{(t+1)}) \ge P(X|θ^{(t)})
$$
现在已知：
$$
Q(θ^{(t+1)},θ^{(t)} \ge Q(θ,θ^{(t)})
$$
而如果想要 $θ^{(t+1)}$使得 $logP(X|θ)$的值有所增加，此时就需要保证：
$$
H(θ^{(t+1)}),θ^{(t)} \le H(θ^{(t)},θ^{(t)})
$$
那么在进行一次迭代之后，$$logP(X|θ)$$的值才会有所增加。
$$
\begin{aligned}
      H(θ^{(t)},θ^{(t)}) - H (θ,θ^{(t)} ) &=\int_Z log  \frac{P(Z|X,θ^{(t)})}{P(Z|X,θ)}P(Z|X,θ^{(t)})dZ \\
      &=\int_Z -log \bigg[\frac{P(Z|X,θ)}{P(Z|X,θ^{(t)})}\bigg]P(Z|X,θ^{(t)})dZ \\
      &= E(-log \bigg[[\frac{P(Z|X,θ)}{P(Z|X,θ^{(t)})} \bigg] \bigg|X,θ^{(t)}) \\
      &\ge -log \bigg[\int_Z\frac{P(Z|X,θ)}{P(Z|X,θ^{(t)})}P(Z|X,θ^{(t)})dZ \bigg]\\
      &= -log\bigg[\int_Z P(Z|X,θ)dZ\bigg] \\
      &= -log (1)\\
      &= 0
      \end{aligned}
$$
  由上述证明可知：
$$
H(θ^{(t)},θ^{(t)}) \ge H(θ,θ^{(t)})
$$
故而有：
$$
H(θ^{(t)},θ^{(t)}) \ge H(θ^{(t+1)},θ^{(t)})
$$
单调递增且有上界，则函数必收敛，现在已经证明单调递增，而$$P(X|θ)$$是一个概率值，必然是小于等于1的，因此最终肯定会收敛。

### 3.3 EM证明数据基础

- 计算期望：

  设随机变量 $$Z$$ 的概率密度函数为 $$f(Z)$$，则 $$E[g(Z)]=\int_Zg(Z)f(Z)dZ$$。

- **琴生（Jensen）不等式（也称为詹森不等式）**

  若 $$f(x)$$是区间 $$[a,b]$$上的下凸函数，则对任意的 $$x_1,x_2,…,x_n\in[a,b]$$且 $$a_1+a_2+,…,+a_n=1,a_1,a_2,…,,a_n$$均为正数，有

  $$
  f(a_1x_1+…,+a_nx_n)\leq a_1f(x_1)+a_2f(x_2)+……+a_nf(x_n)
  $$
  当且仅当$$a_1=a_2=…=a_n=1$$时等号成立。
  
  若 $$f(x)=-ln(x)$$，则有 $$f(E(X)) \leq E(f(x))$$，即 $$ln(E(X))\ge E(ln(x))$$。
  
  证明：
  
  若 $$f(x)$$是凸函数，则有 $$f^{''}(x)\ge 0$$，根据泰勒中值定可知：
  $$
  f(x)=f(x_0)+f^{'}(x_0)(x-x_0)+\frac{1}{2}f^{''}(x_0)(x-x_0)^2
  $$
  由于 $$f^{''}(x)\ge 0$$，则可知：
  $$
  f(x)\ge f(x_0)+f^{'}(x_0)(x-x_0)
  $$
  
  
  令 $$x_0=E(X)$$，则有
  $$
  f(x)\ge f(E(X))+f^{'}(E(X))(x-E(X))
  $$
  不等式两边同时取期望，则有：
  
  $$
  \begin{aligned}E(f(x))&\ge f(E(X))+E(f^{'}(E(X))(x-E(X))) \\ &=f(E(X))+f^{'}(E(X))E((x-E(X)))\\&= f(E(X))\end{aligned}
  $$
  



一个最直观了解 EM 算法思路的是 K-Means 算法。在 K-Means 聚类时，每个聚类簇的质心是隐含数据。我们会假设 K 个初始化质心，即 EM 算法的 E 步；然后计算得到每个样本最近的质心，并把样本聚类到最近的这个质心，即 EM 算法的 M 步。重复这个 E 步和 M 步，直到质心不再变化为止，这样就完成了 K-Means 聚类。








​      

​      

​      

​      

​      

​      

​    

​    

​    

​    







