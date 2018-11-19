=================
CFM
=================


Conjunctive Factor Model:

产生背景：

AFM的一个潜在问题是它处理联合能力点的方式，由于采取加法模型，所以它会倾向将难题预测为简单。所以引出CFM模型，单独计算每个能力点的逻辑概率，
进而将这些概率值进行乘积，达到综合的概率值不大于任意能力点的单独概率值。

算法描述：

CFM的公式定义如下：

.. math::
    p_{ij} = Pr(Y_{ij}=1|\pmb{\theta_i},\pmb{\beta},\pmb{\gamma}) = \prod_{k=1}^K （\frac{exp(\pmb{\theta_i} + \pmb{\beta_k} + \pmb{\gamma_k}T_{ik})}{1 + exp(\pmb{\theta_i} + \pmb{\beta_k} + \pmb{\gamma_k}T_{ik})}) ^ q_{jk}

这里 :math:`Y_{ij}` 是学生i在题目j上的作答结果，:math:`\pmb{\theta_i}` 是学生i的系数，:math:`\pmb{\beta}` 是能力难易系数，
:math:`\pmb{\gamma}` 是能力学习系数，:math:`T_{ik}` 是学生i在能力点k上的练习次数, :math:`q_{jk} = 1` 表示题目j涉及到了能力点k。

根据单数据伯努利分布以及多数据极大似然，可推导出CFM极大似然公式为：

.. math::

    LogLikelihood ll(data) = log(L(data)) = log(\prod_{i=0}^n P_i^{y_i} Q_i^{(1-{y_i})})

    = \sum_{r=1}^n (y_rlog(p_r) + (1-y_r)log(1 - p_r))

接着对上式在 :math:`\pmb{\theta_i}`、:math:`\pmb{\beta_k}`、:math:`\pmb{\gamma_k}` 出求梯度更新对应的值，三式如下：

对于 :math:`\pmb{\theta_i}` 来说，需要找到学生i的所有答题记录，

.. math::

    \frac{dl}{d\pmb{\theta_i}} = \sum_{r=1}^n (\frac{p_r - y_r}{p_r - 1} \sum_{k=1}^K (\frac{1}{1 + e^z_{rk}}))

对于 :math:`\pmb{\beta_k}` 来说，需要找到能力点k的所有答题记录，

.. math::

    \frac{dl}{d\pmb{\beta_k}} = \sum_{i,r \in [1,n]} (\frac{p_r - y_r}{p_r - 1} \frac {e^{k\pmb{\theta_i} + \sum_{k=1}^K \pmb{\beta_k}+ \sum_{k=1}^K \pmb{\gamma_k}T_{ik}}} {\prod_{k=1}^K (1 + e^z_{rk})} \frac {1}{1 + e^z_{rk}})

对于 :math:`\pmb{\gamma_k}` 来说，需要找到能力点k的所有答题记录，

.. math::

    \frac{dl}{d\pmb{\gamma_k}} = \sum_{i,r \in [1,n]} (\frac{p_r - y_r}{p_r - 1} \frac {T_{rk}e^{k\pmb{\theta_i} + \sum_{k=1}^K \pmb{\beta_k}+ \sum_{k=1}^K \pmb{\gamma_k}T_{ik}}} {\prod_{k=1}^K (1 + e^z_{rk})} \frac {1}{1 + e^z_{rk}})