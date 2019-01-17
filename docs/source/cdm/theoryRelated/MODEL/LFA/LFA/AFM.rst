
AFM
=================


Additive Factor Model:

四个假设：

**1. 不同的学生对知识的初始理解不同。** 因此我们为每个学生设置了不同的截距参数。

**2. 学生的学习速率相同。**

**3. 一些项目比其他的要更常见。** 因此我们为每个能力点设置了不同的截距参数。

**4. 一些项目比其他的要容易掌握。** 因此我们为每个能力点设置了不同的斜率参数。

算法描述：

根据上述四个假设，提出了一种多重逻辑回归模型，来对给定能力点的题目反应进行建模，公式如下：

.. math::
    p_{ij} = Pr(Y_{ij}=1|\pmb{\theta_i},\pmb{\beta},\pmb{\gamma}) = \frac{exp(\pmb{\theta_i} + \sum_{k=1}^K q_{jk}(\pmb{\beta_k} + \pmb{\gamma_k}T_{ik}))}{1 + exp(\pmb{\theta_i} + \sum_{k=1}^K q_{jk}(\pmb{\beta_k} + \pmb{\gamma_k}T_{ik}))}

这里 :math:`Y_{ij}` 是学生i在题目j上的作答结果，:math:`\pmb{\theta_i}` 是学生i的系数，:math:`\pmb{\beta}` 是能力难易系数，
:math:`\pmb{\gamma}` 是能力学习系数，:math:`T_{ik}` 是学生i在能力点k上的练习次数, :math:`q_{jk} = 1` 表示题目j涉及到了能力点k。

这个模型通过对题目参数进行线性组合来预测学生的题目作答结果，该模型利用概率p的伯努利分布来进行最大似然估计，该模型在去除掉学习参数时退化为irt形式（但irt应该是没有引入Q矩阵）。

根据单数据伯努利分布以及多数据极大似然，可推导出AFM极大似然公式为：

.. math::

    LogLikelihood ll(data) = log(L(data)) = log(\prod_{i=0}^n P_i^{y_i} Q_i^{(1-{y_i})})

    = \sum_{i=1}^n (y_iz_i - log(1 + e^{z_i}))