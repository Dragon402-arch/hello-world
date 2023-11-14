- 朴素贝叶斯模型：

  首先建模联合概率分布 $P(X,y)$ ，然后使用联合概率分布计算条件概率分布 $P(y|X)$，并依据属性条件独立性假设进行展开 。
  $$
  P(y|X) = \frac{P(X,y)}{P(X)}= \frac{P(y)P(X|y)}{P(X)} = \frac{P(y)}{P(X)} \prod_{i=1}^dP(x_i|y)
  $$
  其中 $P(X|y)$ 是似然，$P(y)$ 是先验概率，$P(y)$ 表达了样本空间中各类别样本所占的比例。
  $$
  y^* = \arg \limits \max \limits_{y} P(y) \prod_{i=1}^dP(x_i|y)
  $$

