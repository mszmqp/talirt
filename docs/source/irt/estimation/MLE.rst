========================
最大似然估计
========================

似然函数
------------------------



单一学生或者单一题目的作答数据的似然函数为

.. math::

    L(\theta;a,b,c|Y) = \prod_{i=0}^n P_i^{y_i} Q_i^{(1-{y_i})}


其中，Y是学生作答结果集合，:math:`y_i` 代表第i条实际作答结果，:math:`P_i` 是预测学生第i条记录作答 **正确** 的概率，
:math:`Q_i=1-P_i` 是预测学生第i条记录作答 **错误** 的概率。
这里， :math:`L(\theta;a,b,c|Y)` 为n个作答样本同时发生的概率。


上述似然函数可以认为是一个学生多条题目的作答记录，或者是，一道题目多个学生的作答记录。

如果扩展到多个学生、多个题目的作答记录，其似然函数为

.. math::

    L(\theta;a,b,c|Y) = \prod_{i=0}^n  \prod_{j=0}^m P_{ij}^{y_{ij}} Q_{ij}^{(1-{y_{ij}})}


其中， :math:`\theta=[\theta_i],a=[a_j],b=[b_j],c=[c_j],\theta,a,b,c` 是向量。

.. note::

    所有学生题目一起估计的似然函数形式，就是把单一学生（题目）独立估计的似然函数连乘在一起，其形式和求解过程是一样的。
    但是注意，学生（题目）独立估计和全部一起估计，估计的结果是不一样的。


上述似然函数有累乘，不利于计算，我们转化为对数函数，

.. math::

    log L(\theta;a,b,c|Y) = \ln L(\theta;a,b,c|Y)

    =\sum_{i=0}^n ({y^{i}} \ln P_i + (1-y^{i}) \ln (1-P_i))



:math:`log L(\theta;a,b,c|Y)` 就是目标函数，问题转化为 **求使目标函数取得极值时参数的值** ，
即使目标函数最大时的参数 :math:`\theta,a,b,c` 的值。

**我们加一个负号，把求最大值问题，转换成求最小值。**

.. math::

    Object function  = - \ln L(\theta;a,b,c|Y)

    =-(\sum_{i=0}^n ({y^{i}} \ln P_i + (1-y^{i}) \ln (1-P_i)))


偏导
=========


一阶偏导
-------------------



.. math::

    \frac{\partial l(\theta) }{\partial \theta}  =\frac{1}{n}\sum_{i=1}^n ( y^{(i)}-\hat{y}^{(i)}) D*a




二阶偏导
-------------------

.. warning::

    在单维IRT模型中，:math:`\theta` 是一个标量，不是向量。
    只有在多维IRT模型中， :math:`\theta` 才是一个向量。而二阶偏导的一般式描述的是参数是向量时的结果。

.. math::

    \text{当}\theta \text{是标量时 }
    H=\frac{\partial^2 J(\theta)}{\partial \theta \partial \theta}=\hat{y} (1-\hat{y}) D^2a^2

    \text{当}\theta \text{是向量时 }
    H_{mn}=\frac{\partial^2 J(\theta)}{\partial \theta_{m} \partial \theta_{n}}=\hat{y} (1-\hat{y}) D^2 a^{(i)}_m a^{(i)}_n










