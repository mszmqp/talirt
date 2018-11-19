PFM
=================


Performance Factor Model:

产生背景：

PFM的产生是用于解决知识追踪（KT）与传统LFA的缺点。

KT的缺点：虽然传统的KT已经以多种形式应用于自适应计算机指导逾40年，但是该模型在域模型的应用困难，且之前没有应用到处理单题目多技能点
情况，同时也没有应用到处理行为的隐藏因素；

传统LFA的缺点：LFA模型并没有跟踪预测每一个学生在每一个技能点的掌握情况进而自适应的进行动态的行为指导（AFM与CFM对联系频率更加敏感，
这可以从LFA公式看出，因此忽略了每个学生的知识点正确错误次数），并且，LFA中假定每个学生的学习速率相同，这也是不科学的。所以将现有模型进行优化来解决上述问题。

算法描述：

PFM依然利用逻辑回归函数形式，只是指数进行了更改，具体指数形式如下：

.. math::
    m(i,j \in KCs, k \in Items, s, f) = \pmb{\beta_k} + \sum_{j \in KCs} (\pmb{\gamma_js_{ij} + p_j\pmb{f_{ij}}})

这里 :math:`\pmb{\beta}` 是题目难易系数，:math:`s_{ij}` 代表了学生i在能力点j上的历史成功次数，:math:`f_{ij}` 是学生i在能力点j上的失败次数, :math:`\pmb{\gamma_j}`, :math:`\pmb{p_j}` 表示了能力点j在成功失败时的调和系数。
模型训练时需要根据 :math:`\pmb{\beta}` 、:math:`\pmb{\gamma_j}` 、:math:`\pmb{p_j}` 来最大化似然函数。

该指数形式中也可以加入 :math:`\pmb{\theta_i}` 来表示学生i的能力值，形式如下：

.. math::
    m(i,j \in KCs, k \in Items, s, f) = \pmb{\theta_i} + \pmb{\beta_k} + \sum_{j \in KCs} (\pmb{\gamma_js_{ij} + p_j\pmb{f_{ij}}})



