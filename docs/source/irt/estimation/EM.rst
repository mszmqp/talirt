=================
EM算法
=================


========================

双参数的IRT模型公式为

.. math::
    P(\theta,a,b) = \frac{1}{1+e^{-Da(\theta-b)}}  \tag{1}

我们设 :math:`\omega_j=[a_j,b_j]^T` 为题目参数，j代表第j个题目；:math:`\theta_i` 为学生潜在能力值，i代表第i个学生。
矩阵 :math:`\mathbf{X}=[X_{ij}]` 为作答记录数据，也就是观测数据。


现在我们的问题是，利用观测数据 :math:`\mathbf{X}` 估计出题目参数 :math:`\pmb{\omega}` 。
根据最大似然估计理论，我们极大化对数似然函数，求解出 :math:`\pmb{\omega}`

.. math::

     \omega = argmax \ lnL(\mathbf{X}|\pmb{\omega})  \tag{2}

但是我们的IRT公式中，除了参数  :math:`\pmb{\omega}` 外， 学生能力参数 :math:`\pmb{\theta}` 也是未知的。
上述似然函数需要转换成边际似然。



.. math::


     lnL(\mathbf{X}|\pmb{\omega}) = ln \sum_{\theta \in \Theta} L(\mathbf{X},\pmb{\theta}|\pmb{\omega}) \tag{3}

.. note::
    在边际化的时候，应该是对 :math:`\pmb{\theta}` 求积分的，假设 :math:`\pmb{\theta}` 是离散随机变量，这样就可以用求和的方式进行积分


公式(3)中存在两个问题。第一，其包含对数中求和的操作，这个基本无法处理；第二，包含另外一个未知参数 :math:`\pmb{\theta}` ，
也就是前文讲的 **缺失数据**  。不要慌，我们可以通过一些变换解决这个问题。

**定理1 Jensen不等式** :
    当 f 是一个下凸函数，并且Y是一个随机变量时，有:

    .. math::
        Ef(Y) \ge f(EY)

    当 f 是一个上凹函数，并且Y是一个随机变量时，有:

    .. math::
        Ef(Y) \le f(EY) \tag{4}

公式(3)是一个上凹函数（有极大值），结合Jensen不等式可以进行一些变换

.. math::

    lnL(\mathbf{X}|\pmb{\omega}) &= \ ln  \sum_{\theta \in \Theta} L(\mathbf{X},\pmb{\theta}|\pmb{\omega})  \\
    &= ln \sum_{\theta \in \Theta} g(\pmb{\theta}) \frac{L(\mathbf{X},\pmb{\theta}|\pmb{\omega})}{g(\pmb{\theta})}  \\
    &\ge \sum_{\theta \in \Theta} g(\pmb{\theta}) ln \frac{L(\mathbf{X},\pmb{\theta}|\pmb{\omega})}{g(\pmb{\theta})}




上式中， :math:`g(\pmb{\theta})` 是 :math:`\pmb{\theta}` 的概率密度（质量）函数，公式第2行，乘以一个 :math:`g(\pmb{\theta})` ，
然后再除以一个 :math:`g(\pmb{\theta})` 等式不变。 :math:`\sum_{\theta \in \Theta} g(\pmb{\theta})` 相当于对后面的
部分 :math:`\frac{L(\mathbf{X},\pmb{\theta}|\pmb{\omega})}{g(\pmb{\theta})}` 求期望，这样结合Jensen不等式得出第三行。

公式(5)提供了似然函数的一个下边界，极大化下边界函数，和极大化似然函数可以得到相似的结果。 :math:`g(\pmb{\theta})` 我们定义成 :math:`\pmb{\theta}` 的
后验概率密度函数  :math:`g(\pmb{\theta} | \mathbf{X},\pmb{\omega})` , 公式(5)就可以转换成

.. math::
    lnL(\mathbf{X}|\pmb{\omega}) = \ ln  \sum_{\pmb{\theta} \in \Theta} L(\mathbf{X},\pmb{\theta}|\pmb{\omega})
    = \sum_{\pmb{\theta} \in \Theta} g(\pmb{\theta} | \mathbf{X},\pmb{\omega})
    ln \frac{L(\mathbf{X},\pmb{\theta}|\pmb{\omega})}{g(\pmb{\theta}|\mathbf{X},\pmb{\omega})}
    \tag{6}

根据贝叶斯定理，:math:`\pmb{\theta}` 的后验概率密度函数可以定义为：

.. math::
     g(\pmb{\theta} | \mathbf{X},\pmb{\omega} ) =
    \frac{L( \mathbf{X}|\pmb{\theta},\pmb{\omega})p(\pmb{\theta},\pmb{\omega})}
    {\sum_{\pmb{\theta}' \in \Theta} L(\mathbf{X}|\pmb{\theta}',\pmb{\omega})p(\pmb{\theta}',\pmb{\omega})  }  \tag{7}


依据对数的性质，公式(6)中的除法可以变成减法

.. math::
    lnL(\mathbf{X}|\pmb{\omega}) =
    \underbrace {\sum_{\pmb{\theta} \in \Theta} g(\pmb{\theta}|\mathbf{X},\pmb{\omega}) ln L(\mathbf{X},\pmb{\theta}|\pmb{\omega})}_{Q(\omega|\omega^{(t)})}
    - \underbrace {\sum_{\pmb{\theta} \in \Theta} g(\pmb{\theta}|\mathbf{X},\pmb{\omega}) ln \ g(\pmb{\theta}|\mathbf{X},\pmb{\omega})}_{H(\omega|\omega^{(t)})}
    \tag{8}


EM算法是一个迭代算法，我们令t表示迭代的序号，t=0,1,2,...。公式(8)中包含两部分， :math:`Q(\pmb{\omega}|\pmb{\omega}^{(t)})`
和 :math:`H(\pmb{\omega}|\pmb{\omega}^{(t)})`
其中 :math:`Q(\pmb{\omega}|\pmb{\omega}^{(t)})` 表示 **完全数据的似然函数** ，也就李航<统计学习>中讲的Q函数。同时，也是在EM算法的M步中要极大化的目标函数。

:math:`H(\pmb{\omega}|\pmb{\omega}^{(t)})` 表示潜在能力变量的后验概率密度（其实是关于后验概率密度函数的一个函数）。
根据Jensen不等式，对于任意 :math:`\pmb{\omega} \in \pmb{\Omega}` ,
:math:`H(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)}) \le H(\pmb{\omega}^{(t)}|\pmb{\omega}^{(t)})` 。

在EM算法的M步骤中，每次迭代 :math:`t \rightarrow t+1` 都是最大化 :math:`Q(\pmb{\omega}|\pmb{\omega}^{(t)})` ，
所以可以确保 :math:`Q(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)}) \ge Q(\pmb{\omega}^{(t)}|\pmb{\omega}^{(t)})` 。

这样，随着每一次迭代，一定能确保对数似然函数的值（我们的目标是极大化对数似然函数）是增长的

.. math::
    &lnL(\mathbf{X}|\pmb{\omega^{(t+1)}}) - lnL(\mathbf{X}|\pmb{\omega^{(t)}}) \\
    &= [Q(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)}) - H(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)})]
        - [ Q(\pmb{\omega}^{(t)}|\pmb{\omega}^{(t)}) - H(\pmb{\omega}^{(t)}|\pmb{\omega}^{(t)})] \\
    &= \underbrace{[Q(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)})- Q(\pmb{\omega}^{(t)}|\pmb{\omega}^{(t)})]}_{\ge0} +
        \underbrace{[H(\pmb{\omega}^{(t)}|\pmb{\omega}^{(t)}) - H(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)})]}_{\ge0} \\
    &\ge 0


以上是简单的推导过程，EM算法的步骤是

- 首先，第一轮迭代t=0时，随机初始化 :math:`\pmb{\omega}^{(t)}`
- E步，求解 :math:`\pmb{\theta}` 的后验概率密度函数 :math:`g(\pmb{\theta} | \mathbf{X},\pmb{\omega}^{(t)} )`
    - :math:`\pmb{\theta}` 本身是连续值，这时  :math:`g(\pmb{\theta} | \mathbf{X},\pmb{\omega}^{(t)} )` 就是概率密度函数，计算积分比较复杂。
    - 所以可以把 :math:`\pmb{\theta}` 离散化，这样 :math:`g(\pmb{\theta} | \mathbf{X},\pmb{\omega}^{(t)} )` 就是概率质量函数，只需要求出其概率分布，然后利用求和的方式计算全概率。
- M步，极大化Q函数 :math:`Q(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)})` 得到新的  :math:`\pmb{\omega}^{(t+1)}`

.. math::
    \pmb{\omega}^{(t+1)} = \mathop{\arg\max}_{\pmb{\omega} \in \pmb{\Omega}}
    \sum_{\pmb{\theta} \in \pmb{\Theta}} g(\pmb{\theta}|\mathbf{X},\pmb{\omega}^{(t)}) ln L(\mathbf{X},\pmb{\theta}|\pmb{\omega})

- 重复E步和M步直到满足收敛条件
    - :math:`\pmb{\omega}` 不再变化 :math:`|\pmb{\omega}^{(t+1)} - \pmb{\omega}^{(t)}|<\epsilon`
    - 对数似然函数不再变化 :math:`|lnL(\mathbf{X}|\pmb{\omega}^{(t+1)}) - lnL(\mathbf{X}|\pmb{\omega}^{(t)})|<\epsilon`

.. math::

    \omega^{t+1} = argmax \sum_{j=1}^J \sum_{k=1}^{K} \hat{r}_{jk} ln P_{w_j}(\theta_k) + \hat{W}_{jk} ln (1- P_{w_j}(\theta_k))



似然函数
========================





.. math::

    L(a,b|Y,\theta) = \prod_{i=1}^n \prod_{j=1}^m P_{ij}^{y_{ij}} Q_{ij}^{(1-{y_{ij}})}


其中，Y是学生作答结果集合，:math:`y_{ij}` 代表第i个学生作答第j个题目的结果，:math:`P_{ij}` 是预测第i个学生作答第j个题目 **正确** 的概率，
:math:`Q_{ij}=1-P_{ij}` 是预测第i个学生作答第j个题目 **错误** 的概率。
这里， :math:`L(a,b|Y,\theta)` 为n*m个作答样本同时发生的概率。



上述似然函数有连乘，不利于计算，我们转化为对数函数，

.. math::

    log L(a,b|Y,\theta) = \ln L(\theta;a,b|Y)

    =\sum_{i=1}^n \sum_{j=1}^m ({y^{ij}} \ln P_{ij} + (1-y^{ij}) \ln (1-P_{ij}))



:math:`logL(a,b|Y,\theta)` 就是目标函数，问题转化为 **求使目标函数取得极值时参数的值** ，
即使目标函数最大时的参数 :math:`a,b` 的值。

**我们加一个负号，把求最大值问题，转换成求最小值。**

.. math::

    Object\ function \quad  l(a,b|Y,\theta) = - \ln L(a,b|Y,\theta)

    =-(\sum_{i=1}^n \sum_{j=1}^m ({y^{ij}} \ln P_{ij} + (1-y^{ij}) \ln (1-P_{ij})))

独立性
===================

根据独立性假设，每个题目独立进行估计

偏导
==================


一阶导数（Jacobian矩阵）
---------------------------

.. math::

    \frac{\partial l(a,b|Y,\theta)} {\partial a}  = \sum_{i=1}^n ( y-\hat{y}) \theta

    \frac{\partial l(a,b|Y,\theta) }{\partial b}  = \sum_{i=1}^n ( y-\hat{y})


二阶导数（Hessian矩阵）
---------------------------

.. math::

    \frac{\partial^2 l(a,b|Y,\theta)} {\partial a^2}  = \sum_{i=1}^n \hat{y} ( 1 - \hat{y}) \theta^2

    \frac{\partial^2 l(a,b|Y,\theta) }{\partial b^2}  = \sum_{i=1}^n \hat{y} ( 1 - \hat{y})

     \frac{\partial^2 l(a,b|Y,\theta) }{\partial ab}  = \sum_{i=1}^n \hat{y} ( 1 - \hat{y})\theta




.. math::

    H(a,b|Y,\theta)=\left\{
    \begin{aligned}
    \frac{\partial^2 l(a,b|Y,\theta)} {\partial a^2},\ \frac{\partial^2 l(a,b|Y,\theta) }{\partial ab}

    \frac{\partial^2 l(a,b|Y,\theta) }{\partial ab} ,\ \frac{\partial^2 l(a,b|Y,\theta) }{\partial b^2}
    \end{aligned}
    \right\}.


得到了 :math:`\alpha` 的基础上，然后可以得到 :math:`\beta_m` 。

.. math::
    \beta_m = arg \ min \sum_{i=1}^N L(y_i,F_{m-1}(x_i) + \beta h(x_i;\alpha_m))







参考内容
===============================================
　　[1] `IRT Parameter Estimation using the EM Algorithm <http://www.openirt.com/b-a-h/papers/note9801.pdf>`_

　　[2] `RoutledgeHandbooks-9781315736013-chapter3 <https://www.routledgehandbooks.com/doi/10.4324/9781315736013.ch3>`_

　　[3] `Optimizing Information Using the Expectation-Maximization Algorithm in Item Response Theory <https://www.lsac.org/docs/default-source/research-(lsac-resources)/rr-11-01.pdf?sfvrsn=2>`_


