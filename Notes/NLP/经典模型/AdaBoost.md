AdaBoost算法的特点是通过迭代每次学习一个基本分类器。每次迭代中，提高那些被前一轮分类器错误分类的样本权重值，而降低那些被正确分类的样本权值。最后，AdaBoost将基本分类器的线性组合作为最终分类器，并且对于分类误差较小的基本分类器给予较高的系数，对于分类误差较大的基本分类器给予较小的系数。

AdaBoost可以看做是前向分布算法的一个实现。

### 1. AdaBoost 算法

给定训练数据集 $ \{ (x_i,y_i)\}_{i=1}^n$，其中 $x_i \in R^d,y \in \{-1,1\}$，训练轮数为 $T$

1. 初始化训练样本的权值：$D_1 = (w_{11},w_{12},…,w_{1n}),\large w_{1i}=\frac{1}{n}  \text{      ,        } \forall i = 1,2…,n  $ 

2. 对于 $t = 1,2,…,T$ 

   - 使用具有样本权值分布 $D_t$ 的训练数据集学习，得到基本分类器 
     $$
     h_t(x):x \rightarrow \{-1,+1\}
     $$

   - 计算 $h_t(x)$ 在训练数据集上的分类误差率：
     $$
     e_t = \sum_{i=1}^nP(h_t(x_i)\not=y_i) = \sum_{h(x_i)\not=y_i} w_{ti}=  \sum_{i=1}^nw_{ti}I(h_t(x_i)\not=y_i)
     $$

   - 计算 $h_t(x)$ 的系数 $\alpha_t$：
     $$
     \alpha_t = \frac{1}{2} \log \frac{1-e_t}{e_t} =  \frac{1}{2} \ln \frac{1-e_t}{e_t}
     $$

   - 更新训练数据样本权值：
     $$
     \begin{align}
     \large D_{t+1} &= (w_{t+1,1},w_{t+1,2},…,w_{t+1,n}) \\
     \large w_{t+1,i} &= \frac{w_{t,i}·exp\big(-\alpha_th_t(x_i)y_i\big)}{Z_t}   \text{      ,        } \forall i = 1,2…,n
     \end{align}
     $$
     其中 $Z_t$ 是规范化因子，可使得 $D_{t+1}$ 成为一个概率分布，即 $\large \sum_{i=1}^nw_{t,i} = 1$。
     $$
     Z_t = \sum_{i=1}^nw_{t,i}·exp\big(-\alpha_th_t(x_i)y_i\big)
     $$

3. 构建基本分类器的线性组合，并输出最终分类器
   $$
   H(x) = \text{ sign}\bigg(\sum_{t=1}^T \alpha_th_t(x)\bigg)
   $$



注解：

- 当 $e_t\le \frac{1}{2}$ 时，有 $\alpha_t \ge 0$，且 $\alpha_t$ 随着 $e_t$ 的减小而增大，所以分类误差较小的分类器在最终分类器中的作用更大。

- 样本权值更新公式可以写成：
  $$
  \large w_{t+1,i} = \begin{cases}
  \frac{w_{t,i}}{Z_t}e^{-\alpha_t}, &  h_t(x_i)=y_i \\\\\
  \frac{w_{t,i}}{Z_t}e^{\alpha_t}, &  h_t(x_i)\not=y_i
  \end{cases}
  $$
  由上面的公式可知，分类错误的样本权值将会放大，而分类正确的样本权值则会缩小。同一个样本，分类错误后的权值是分类正确的权值的 $e^{2\alpha_t}$ 倍。因此，分类错误的样本在下一轮学习中将会受到更多的关注。

- 关于 $\alpha$ 的取值，并不一定满足 $\sum_{t=1}^T\alpha_t = 1$ 。

​	



### 2. AdaBoost 算法推导

假定已知 $h_1(x),h_2(x),…,h_{t-1}(x)$，以及其对应的系数 $\alpha_1,\alpha_2,…,\alpha_{t-1}$ ，想要求出 $h_t(x),\alpha_t$，于是有：
$$
\alpha_t,h_t(x) =\arg \limits \min \limits_{\alpha,h} \sum_{i=1}^nL(y_i,H_{t-1}(x_i)+\alpha h(x_i))
$$
其中 $H_{t-1}(x)$
$$
H_{t-1}(x) = \sum_{i=1}^{t-1}\alpha_ih_i(x)
$$
损失函数使用指数损失函数（exponential loss）：
$$
L(y,\hat y) = \large e^{-y\hat y}
$$
将损失函数带入目标函数则有：
$$
\begin{align}
\alpha_t,h_t(x) &=\large \arg \limits \min \limits_{\alpha,h} \sum_{i=1}^nL(y_i,H_{t-1}(x_i)+\alpha h(x_i)) \\
 &=\large \arg \limits \min \limits_{\alpha,h} \sum_{i=1}^n e^{-y_i\big(H_{t-1}(x_i)+\alpha h(x_i)\big)} \\
  &=\large \arg \limits \min \limits_{\alpha,h} \sum_{i=1}^n e^{-y_i\alpha h(x_i)} e^{-y_iH_{t-1}(x_i)}
\end{align}
$$
其中 $\large e^{-y_iH_{t-1}(x_i)} $ 是第 $t$  轮学习时是一个与 $\alpha,h$ 无关的已知常数，与最小化结果无关，但其会随着每一轮迭代而发生变化。

令 $ \large \beta_{ti} = e^{-y_iH_{t-1}(x_i)} $，然后可以按照正确分类样本点和错误分类样本点拆分求和项，则有：
$$
\begin{align}
\alpha_t,h_t(x) 
  &=\large \arg \limits \min \limits_{\alpha,h} \sum_{i=1}^n e^{-y_i\alpha h(x_i)} e^{-y_iH_{t-1}(x_i)}\\
  &=\large \arg \limits \min \limits_{\alpha,h} \sum_{i=1}^n \beta_{ti}  e^{-y_i\alpha h(x_i)} \\
  &=\large \arg \limits \min \limits_{\alpha,h} \sum_{h(x_i)=y_i} \beta_{ti}  e^{-\alpha} +  \sum_{h(x_i)\not=y_i} \beta_{ti}  e^{\alpha}  \\
  &=\large \arg \limits \min \limits_{\alpha,h}  e^{-\alpha}(\sum_{i=1}^n \beta_{ti} -\sum_{h(x_i)\not=y_i}\beta_{ti}) +  \sum_{h(x_i)\not=y_i} \beta_{ti}  e^{\alpha}  \\
  &=\large \arg \limits \min \limits_{\alpha,h}  (e^{\alpha} -e^{-\alpha})\sum_{h(x_i)\not=y_i} \beta_{ti} +e^{-\alpha}\sum_{i=1}^n \beta_{ti} \\
  
\end{align}
$$
首先，求 $h_t(x)$。对于任意 $\alpha > 0$ （也即是在$\alpha$ 给定的情况下），最后一行表达式的第二项与 $h_t(x)$ 无关（是上一轮迭代后可以计算的一个结果，是个常数），在去除已知常数后，使得目标函数取值最小的 $h_t(x)$ 由下面的式子得到：
$$
h_t^*(x) = \large \arg \limits \min \limits_{h} \sum_{h(x_i)\not=y_i} \beta_{ti} = \arg \limits \min \limits_{h} \sum_{i=1}^n \beta_{ti}I(h(x_i)\not=y_i)
$$
可以看到 $h_t(x) $ 的最优选择是使第 $t$ 轮错误分类样本的总权重和最小的分类器。

然后，求解 $\alpha_t$。

设置：
$$
\large e_t =\frac{\sum_{h^*(x_i)\not=y_i} \beta_{ti}}{\sum_{i=1}^n \beta_{ti}}
$$
将已经求得的 $h_t^*(x)$ 带入目标函数，并将 $e_t$ 带入目标 函数中则有：
$$
\begin{align}
\alpha_t
  &=\large \arg \limits \min \limits_{\alpha}  (e^{\alpha} -e^{-\alpha})\sum_{h^*(x_i)\not=y_i} \beta_{ti} +e^{-\alpha}\sum_{i=1}^n \beta_{ti} \\
   &=\large \arg \limits \min \limits_{\alpha}  (e^{\alpha} -e^{-\alpha})e_t\sum_{i=1}^n \beta_{ti}  +e^{-\alpha}\sum_{i=1}^n \beta_{ti} \\
 &=\large \sum_{i=1}^n \beta_{ti} \arg \limits \min \limits_{\alpha}  (e^{\alpha} -e^{-\alpha})e_t  +e^{-\alpha} \\
  &=\large \arg \limits \min \limits_{\alpha}  (e^{\alpha} -e^{-\alpha})e_t  +e^{-\alpha} \\
    &=\large \arg \limits \min \limits_{\alpha}  e^{\alpha}e_t+(1-e_t)e^{-\alpha} \\
\end{align}
$$
对 $\alpha$ 求导并令其等于0，则有：
$$
\begin{align} 0 &= −(1 − e_t)e^{−α} + e^αe_t = e_t-1 + e^{2α}e_t \\
\end{align}
$$
可得：
$$
\begin{align}
e^{2\alpha_t} &= \frac{1-e_t}{e_t} \\
\alpha_t &= \frac{1}{2}\ln \frac{1-e_t}{e_t}
\end {align}
$$
最后确定每一轮样本权重的更新，由 
$$
\begin{align}
H_t(x) &= H_{t-1}(x) + \alpha_th_t(x) 
\end {align}
$$
以及
$$
\large \beta_{ti} = \large e^{-y_iH_{t-1}(x_i)}
$$
可得：
$$
\large \beta_{t+1,i} = \large e^{-y_iH_t(x_i)} =  \large e^{-y_i( H_{t-1}(x) + \alpha_th_t(x) )} =  \beta_{ti}e^{-y_i\alpha_th_t(x)}
$$
为使得样本权重分布为概率分布，也就是使得 $\sum_i^n\beta_{t,i} = 1$，需要对上式中的样本权重进行归一化操作。
$$
\large \beta_{t+1,i} =  \frac {\beta_{ti}e^{-y_i\alpha_th_t(x)}}{\sum_{i=1}^n{\beta_{ti}e^{-y_i\alpha_th_t(x)}}}
$$

### 3 AdaBoost 使用示例

参考李航《统计学习方法》P159，计算过程如下：

```python
#  -*- coding: utf-8 -*-
import math

train_data = {0: 1, 1: 1, 2: 1, 3: -1, 4: -1, 5: -1, 6: 1, 7: 1, 8: 1, 9: -1}
weights = [round(1 / len(train_data), 5)] * 10

# print(weights)

spots = [i + 0.5 for i in range(9)]


# print(spots)


def calculate_error_and_preds(spot, operator="<="):
    preds = []
    total_error = 0.0
    for idx, item in enumerate(train_data.items()):
        key, value = item
        if operator == "<=":
            pred = 1 if key <= spot else -1
        else:
            pred = 1 if key >= spot else -1
        preds.append(pred)
        total_error += (1 if pred != value else 0) * weights[idx]
    return round(total_error, 5), preds


def get_opt_spot(spots):
    le_errors = [calculate_error_and_preds(spot, operator="<=")[0] for spot in spots]
    # print(le_errors)
    min_le_error = min(le_errors)

    ge_errors = [calculate_error_and_preds(spot, operator=">=")[0] for spot in spots]
    # print(ge_errors)
    min_ge_error = min(ge_errors)

    min_error = min(min_ge_error, min_le_error)
    if min_error == min_ge_error:
        opt_spot = spots[ge_errors.index(min_error)]
        operator = ">="
    else:
        opt_spot = spots[le_errors.index(min_error)]
        operator = "<="

    return min_error, opt_spot, operator


def get_calculated_elements(spots):
    min_error, opt_spot, operator = get_opt_spot(spots)

    preds = calculate_error_and_preds(opt_spot, operator)[1]

    alpha = round(0.5 * math.log((1 - min_error) / min_error), 5)
    return preds, alpha, min_error, opt_spot


def update_weights(round_idx, weights, spots):
    preds, alpha, min_error, opt_spot = get_calculated_elements(spots)
    weights = [
        weight * math.exp(-1 * alpha * pred * y)
        for weight, y, pred in zip(weights, train_data.values(), preds)
    ]
    total_sum = sum(weights)
    weights = [round(weight / total_sum, 5) for weight in weights]
    print(f"第{round_idx}轮最优划分点为{opt_spot},对应的误差为{min_error},alpha取值为{alpha},权重分布为{weights}")
    return weights


for round_idx in range(1, 4):
    weights = update_weights(round_idx, weights, spots)
    
"""
输出：
第1轮最优划分点为2.5,对应的误差为0.3,alpha取值为0.42365,权重分布为[0.07143, 0.07143, 0.07143, 0.07143, 0.07143, 0.07143, 0.16667, 0.16667, 0.16667, 0.07143]
第2轮最优划分点为8.5,对应的误差为0.21429,alpha取值为0.64963,权重分布为[0.04546, 0.04546, 0.04546, 0.16666, 0.16666, 0.16666, 0.10606, 0.10606, 0.10606, 0.04546]
第3轮最优划分点为5.5,对应的误差为0.18184,alpha取值为0.75197,权重分布为[0.125, 0.125, 0.125, 0.10185, 0.10185, 0.10185, 0.06482, 0.06482, 0.06482, 0.125]
"""

```

