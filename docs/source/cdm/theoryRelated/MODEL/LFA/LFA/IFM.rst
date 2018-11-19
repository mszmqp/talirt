IFM
=================

Instructional Factors Analysis:

产生背景：

IFM的产生是用于传统AFM、PFM在现实应用中的缺点，因为在现实自适应教学系统中，并不全是基于学生驱动的，有些数据既包含了学生驱动，也包含了多种类型的
指导干涉，并且这些指导并没有直接作用于可观察到的学生行为上，IFM便是在AHM、PFM的基础上添加了学生在知识点上的获得指导的次数。

算法描述：

IFM依然利用逻辑回归函数形式，只是指数进行了更改，具体指数形式如下：

.. math::
    m(i,j \in KCs, k \in Items, s, f) = \pmb{\theta_i} + \pmb{\beta_k} + \sum_{j \in KCs} (\pmb{\gamma_js_{ij}} + \pmb{p_j}\pmb{f_{ij}} + \pmb{v_j}\pmb{T_{ij}})

这里 :math:`\pmb{\theta_i}` 来表示学生i的能力值，:math:`\pmb{\beta}` 是题目难易系数，:math:`s_{ij}` 代表了学生i在能力点j上的历史成功次数，:math:`f_{ij}` 是学生i在能力点j上的失败次数,
:math:`\pmb{\gamma_j}`, :math:`\pmb{p_j}` 表示了能力点j在成功失败时的调和系数, :math:`v_j` 表示在能力点j上的指导增益，:math:`T_{ij}` 表示学生i在能力点j上接受的指导次数，
模型训练时需要根据 :math:`\pmb{\beta}` 、:math:`\pmb{\gamma_j}` 、:math:`\pmb{p_j}` 来最大化似然函数。




