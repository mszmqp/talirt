=================
EM算法
=================

EM算法简介
========================

双参数的IRT模型公式为

.. math::
    P(\theta,a,b) = \frac{1}{1+e^{-Da(\theta-b)}}  \tag{1}

我们设 :math:`\omega_j=[a_j,b_j]^T` 为题目参数，j代表第j个题目；:math:`\theta_i` 为学生潜在能力值，i代表第i个学生。
矩阵 :math:`\mathbf{X}=[X_{ij}]` 为作答记录数据，也就是观测数据。


现在我们的问题是，利用观测数据 :math:`\mathbf{X}` 估计出题目参数 :math:`\pmb{\omega}` 。
根据最大似然估计理论，我们极大化对数似然函数，求解出 :math:`\pmb{\omega}`

.. math::

     \omega = \mathop{\arg\max} \ lnL(\mathbf{X}|\pmb{\omega})  \tag{2}

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

.. note::
    :math:`\pmb{\theta}` 是学生能力值，我们看作一个随机变量。既然是随机变量就会有其概率分布。
    对于连续值随机变量，其概率分布函数称为概率密度函数， **函数值不是概率值** ，需要进行积分才能得出概率值（不懂的，去恶补概率论基础）；
    对于离散值随机变量，其概率分布函数称为概率质量函数， **函数值就是概率值** 。
    公式中的 :math:`g(\pmb{\theta} | \mathbf{X},\pmb{\omega})` 就是随机变量 :math:`\pmb{\theta}` 的概率分布函数。
    当然，这里 :math:`\pmb{\theta}` 学生能力值应该是连续值，但是积分操作过于复杂，所以我们可以把 :math:`\pmb{\theta}` 进行离散化，
    把 :math:`\pmb{\theta}` 转换成离散值，这样就不用积分操作了。离散化的方法以及 :math:`g(\pmb{\theta} | \mathbf{X},\pmb{\omega})` 如何求下文介绍。



随机变量 :math:`\pmb{\theta}` 概率分布函数是什么呢？ 我们既然已经有了学生的作答记录，
那么，我们可以利用学生的作答记录来计算 :math:`\pmb{\theta}` 的后验概率分布。所以公式(6)中使用的是 :math:`\pmb{\theta}` 的后验概率。
根据贝叶斯定理，:math:`\pmb{\theta}` 的后验概率密度函数可以定义为：

.. math::
     g(\pmb{\theta} | \mathbf{X},\pmb{\omega} ) =
    \frac{L( \mathbf{X}|\pmb{\theta},\pmb{\omega})p(\pmb{\theta},\pmb{\omega})}
    {\sum_{\pmb{\theta}' \in \Theta} L(\mathbf{X}|\pmb{\theta}',\pmb{\omega})p(\pmb{\theta}',\pmb{\omega})  }  \tag{7}

.. note::
    什么是先验概率、后验概率，这里不介绍了，自己恶补概率论基础吧。
    根据贝叶斯定理，公式(7)的分母部分应该是全概率公式，如果 :math:`\pmb{\theta}` 是连续值，这里是要对 :math:`\pmb{\theta}` 进行积分的。
    之前我们说过，我们会对 :math:`\pmb{\theta}`  进行离散化，这样就可以把积分操作转换成求和操作。


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
    &lnL(\mathbf{X}|\pmb{\omega}^{(t+1)}) - lnL(\mathbf{X}|\pmb{\omega}^{(t)}) \\
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
    \tag{10}

- 重复E步和M步直到满足收敛条件
    - :math:`\pmb{\omega}` 不再变化 :math:`|\pmb{\omega}^{(t+1)} - \pmb{\omega}^{(t)}|<\epsilon`
    - 对数似然函数不再变化 :math:`|lnL(\mathbf{X}|\pmb{\omega}^{(t+1)}) - lnL(\mathbf{X}|\pmb{\omega}^{(t)})|<\epsilon`



IRT项目参数估计
========================

独立性假设
------------------

首先说明一些假设

**假设1，学生的作答行为是相互独立事件**，在作答题目时，学生与学生之间互不影响。

作答记录 :math:`\mathbf{X}_r` 和  :math:`\mathbf{X}_s` 分别是学生r、学生s的作答记录向量，
那么， :math:`L(\mathbf{X}_r,\mathbf{X}_s|\pmb{\omega})=L(\mathbf{X}_r|\pmb{\omega})L(\mathbf{X}_s|\pmb{\omega})` 。
因此，对数似然函数可以改写成

.. math::
    ln L(\mathbf{X}|\pmb{\omega}) = ln \prod_{i=1}^N L(\mathbf{X}_i|\pmb{\omega}) = \sum_{i=1}^N ln L(\mathbf{X}_i|\pmb{\omega})
    \tag{11}

其中i=1,2,3,...,N,i代表学生编号, :math:`\mathbf{X}_i` 代表学生i作答 :math:`\mathbf{J}` 个题目的作答向量，题目的参数向量是 :math:`\pmb{\omega}`


结合公式(11)和公式(6),对数似然函数可以变换为:

.. math::
    ln L(\mathbf{X}|\pmb{\omega}) = \sum_{i=1}^N ln L(\mathbf{X}_i|\pmb{\omega})
    = \sum_{i=1}^N  \sum_{\pmb{\theta} \in \Theta} g(\pmb{\theta} | \mathbf{X}_i,\pmb{\omega})
    ln \frac{L(\mathbf{X}_i,\pmb{\theta}|\pmb{\omega})}{g(\pmb{\theta}|\mathbf{X}_i,\pmb{\omega})}
    \tag{12}




**假设2，题目的作答是相互独立事件** ，学生在作答当前题目时，不受其它题目作答结果的影响。

.. math::
    L(\mathbf{X}_i,\pmb{\theta}|\pmb{\omega})=L(\mathbf{X}_i|\pmb{\theta},\pmb{\omega}) p(\pmb{\theta}|\pmb{\omega})
    =\prod_{j=1}^J L(\mathbf{X}_{ij}|\pmb{\theta},\pmb{\omega}_j) p(\pmb{\theta}|\pmb{\omega}_j)
    \tag{13}

公式(13)表达的是一个学生的J个题目对数似然函数。

**假设3，题目参数和学生能力值是相互独立的** ， :math:`p(\pmb{\theta},\pmb{\omega})=p(\pmb{\theta})p(\pmb{\omega})` ,
并且，:math:`p(\pmb{\theta}|\pmb{\omega})=p(\pmb{\theta})`
这里 :math:`p(\pmb{\theta})` 和 :math:`p(\pmb{\omega})` 是先验概率

依此，公式(13)可以变换成

.. math::
    L(\mathbf{X}_i,\pmb{\theta}|\pmb{\omega}) = =\prod_{j=1}^J L(\mathbf{X}_{ij}|\pmb{\theta},\pmb{\omega}_j) p(\pmb{\theta})
    \tag{14}


**假设4，学生能力值是单维变量**

好像是废话，IRT是分为单维能力模型和多维能力模型的，本篇讲的的单维IRT模型，也就是对于一个学生来说，其能力值 :math:`\theta` 是一个标量值。
事实上，EM算法也只能解决单维IRT模型的参数估计。

能力值积分点
-----------------------

前文多次提到，随机变量 :math:`\theta` 是连续值，需要进行积分，不利于我们计算，所以我们可以把它进行离散化，方便我们计算其概率分布。
假设 :math:`\theta` 的取值空间是 :math:`\Theta` ,空间 :math:`\Theta` 是实数值空间，
我们从空间 :math:`\Theta` 取K个值，作为 :math:`\theta` 的取值，也就是我们把 :math:`\theta` 从无限的取值空间强制变成只能取K个值。
:math:`\theta_k` 表示 :math:`\theta` 的第k个取值。

结合前文的推到以及假设，对数似然函数可以进行如下变换:

.. math::
    ln L(\mathbf{X}|\pmb{\omega}) &= \sum_{i=1}^N ln L(\mathbf{X}_i|\pmb{\omega}) \\
    &= \sum_{i=1}^N \sum_{k=1}^K g(\theta_k|\mathbf{X}_i,\pmb{\omega})
        ln \frac{L(\mathbf{X}_i,\theta_k|\pmb{\omega})}{g(\theta_k|\mathbf{X}_i,\pmb{\omega})} \\
    &= \sum_{i=1}^N \sum_{k=1}^K g(\theta_k|\mathbf{X}_i,\pmb{\omega})
        ln \frac{\prod_{j=1}^J L(X_{ij}|\theta_k,\pmb{\omega}_j)p(\theta_k)} {g(\theta_k|\mathbf{X}_i,\pmb{\omega})}
        \ \ \text{根据假设3} \\
    &= \underbrace{\sum_{i=1}^N \sum_{k=1}^K g(\theta_k|\mathbf{X}_i,\pmb{\omega})
        \left [ \sum_{j=1}^J ln\ L(X_{ij}|\theta_k,\pmb{\omega}_j)+ ln \ p(\theta_k) \right ] }_{Q} \\
    &- \underbrace{\sum_{i=1}^N \sum_{k=1}^K g(\theta_k|\mathbf{X}_i,\pmb{\omega}) ln\ g(\theta_k|\mathbf{X}_j,\pmb{\omega})}_{H}
    \qquad \qquad \text{(15)}

最终转换成前文公式(8)里的 :math:`Q-H` 的模式。根据前文的推导，我们需要极大化Q函数部分，求得题目参数 :math:`\pmb{\omega}` ,但是Q函数中包含三部分

.. math::

    Q=\sum_{i=1}^N \sum_{k=1}^K \left [ \underbrace{g(\theta_k|\mathbf{X}_i,\pmb{\omega})}_{\text{能力值后验概率}}
         \left [ \sum_{j=1}^J \underbrace{ln\ L(X_{ij}|\theta_k,\pmb{\omega}_j)}_{\text{对数似然函数}}+
          \underbrace{ ln \ p(\theta_k)}_{\text{能力值对数先验分布}} \right ] \right ]  \tag{16}

在EM算法的M步，每一轮迭代极大化Q函数，求得题目参数 :math:`\pmb{\omega}`

.. math::
    \pmb{\omega}^{(t+1)} &= \mathop{\arg\max}_{\pmb{\omega} \in \Omega} \sum_{i=1}^N \sum_{k=1}^K
    g(\theta_k|\mathbf{X}_i,\pmb{\omega}^{(t)}) ln L(\mathbf{X}_i,\theta_k|\pmb{\omega}) \\
    &\Leftrightarrow  \mathop{\arg\max}_{\pmb{\omega} \in \Omega} \sum_{i=1}^N \sum_{k=1}^K g(\theta_k|\mathbf{X}_i,\pmb{\omega}^{(t)})
    \left [  \sum_{j=1}^J  ln\ L(X_{ij}|\theta_k,\pmb{\omega}_j) + ln \ p(\theta_k) \right ]
    \qquad \qquad \text{(17)}


先验概率分布
----------------------
从经验上来讲，我们一般认为学生能力值是服从正态分布的，所以我们可以假设 :math:`\theta` 的先验是服从标准正态分布，这样其先验分布就是标准正态分布。
前文讲过，我们通过把 :math:`\theta` 离散化，解决积分的问题。

假设 :math:`\theta` 的取值范围是 [-6,6],从中均匀取出n=40个点，作为积分点，也就是假设 :math:`\theta` 只能取这n个值。

.. code-block:: python

    theta_min = kwargs.get('theta_min', -6)
    theta_max = kwargs.get('theta_max', 6)
    theta_num = kwargs.get('theta_num', 40)
    theta_distribution = kwargs.get('theta_distribution', 'normal')

    self.Q = theta_num
    # 从指定区间等距离取出点，作为先验分布的积分点
    self.theta_prior_value = np.linspace(theta_min, theta_max, num=theta_num)

    if self.Q != len(self.theta_prior_value):
        raise Exception('wrong number of inintial theta values')
    # 先验分布是均匀分布
    if theta_distribution == 'uniform':
        self.theta_prior_distribution = np.ones(self.Q) / self.Q
    # 先验分布是标准正态分布
    elif theta_distribution == 'normal':
        norm_pdf = [norm.pdf(x) for x in self.theta_prior_value]
        normalizer = sum(norm_pdf)
        self.theta_prior_distribution = np.array([x / normalizer for x in norm_pdf])
    else:
        raise Exception('invalid theta prior distribution %s' % theta_distribution)
    # theta后验分布初始值
    self.theta_posterior_distribution = np.zeros((self.user_count, self.Q))



对数似然函数
----------------------


一条答题记录的似然函数是：

.. math::
    ln L(\mathbf{X}_{ij}=x_{ij}|\theta_k,\pmb{\omega}_j) &=
    ln \left [  P_{\omega_j}(\theta_k)^{X_{ij}} \left(1-P_{\omega_j}(\theta_k)\right)^{1-X_{ij}}  \right ] \\
    &= X_{ij} ln P_{\omega_j}(\theta_k) + (1-X_{ij})ln \left(1-P_{\omega_j}(\theta_k)\right)
    \qquad \qquad \text{(18)}

把公式(18) 带入 公式(17) 得,注意公式(17)中的 :math:`ln\ p(\theta_k)` 部分是可以省略的，由于其不是关于 :math:`\pmb{\omega}` 的函数，
在极大化过程中可以省略。

.. math::
    \pmb{\omega}^{(t+1)} = \mathop{\arg\max}_{\pmb{\omega} \in \Omega} \sum_{j=1}^J \sum_{i=1}^N \sum_{k=1}^K g(\theta_k|\mathbf{X}_i,\pmb{\omega}^{(t)})
    \left[   X_{ij} ln P_{\omega_j}(\theta_k) + (1-X_{ij})ln \left(1-P_{\omega_j}(\theta_k)\right) \right] \tag{19}


后验概率分布
----------------------

其中，能力值 :math:`\theta` 的后验概率分布，求后验概率分布时， :math:`\pmb{\omega}` 的值认为是已知的，
用的上一轮的估计值。 至于 :math:`\pmb{\theta}` 自然是只能取先验中固定的值。

在实际计算的过程中，公式中的 :math:`\prod_{j=1}^J L(X_{ij}|\theta_k,\pmb{\omega}_j)` 部分存在连乘，
所以会先加一个对数操作，算完后再用指数计算得到正确值。



.. math::

    g(\theta_k|\mathbf{X}_i,\pmb{\omega}) = \frac{\prod_{j=1}^J L(X_{ij}|\theta_k,\pmb{\omega}_j) p(\theta_k)}
    {\sum_{m=1}^K \prod_{j=1}^J L(X_{ij}|\theta_m,\pmb{\omega}_j) p(\theta_m)}



.. code-block:: python


        def _update_posterior_distribution(self):
        """
        计算每个学生的后验概率分布
        self.theta_prior_distribution 是 theta的先验概率分布
        self.theta_posterior_distribution theta的后验概率分布
        Returns
        -------

        """
            def logsum(logp: np.ndarray):
                """
                后验概率的分母部分的计算。
                注意是加了对数的。
                """
                w = logp.max(axis=1)
                shape = (w.size, 1)
                w = w.reshape(shape)
                logSump = w + np.log(np.sum(np.exp(logp - w), axis=1)).reshape(shape)
                return logSump

            # self.Q 是能力值theta离散化的取值数量
            for k in range(self.Q):
                # 对于theta的每一个可能取值都进行

                # 假设每个学生的能力值都是theta_k
                theta_k = np.asarray([self.theta_prior_value[k]] * self.user_count).flatten()
                # theta取值为theta_k的先验概率
                theta_k_prior_prob = self.theta_prior_distribution[k]
                # 每个学生独立计算，各自作答数据的的log似然值。
                # 注意实际公式中是连乘符号，乘法会造成小数溢出，所以我们计算其对数值，把乘法转换成加法，注意最后还得换回去
                independent_user_lld = uirt_clib.log_likelihood_user(response=self.response,
                                                                     theta=theta_k,
                                                                     slope=self.a,
                                                                     intercept=self.b, guess=self.c)
                # 乘上当theta值的先验概率,这是后验概率分布公式中的分子
                self.theta_posterior_distribution[:, k] = independent_user_lld + np.log(theta_k_prior_prob)
            # 上述循环，计算出了theta每个取值theta_k的分子部分
            # 后验概率的分母不是很好求
            # 后验概率分布更新，分子减分母，差值再求自然指数
            self.theta_posterior_distribution = np.exp(
                self.theta_posterior_distribution - logsum(self.theta_posterior_distribution))
            # 检查后验概率分布的概率和是否为1
            self.__check_theta_posterior()
            return 1


.. math::

    \omega^{t+1} = \mathop{\arg\max}_{\pmb{\omega} \in \pmb{\Omega}} \sum_{j=1}^J \sum_{k=1}^{K} \hat{r}_{jk} ln P_{w_j}(\theta_k) + \hat{W}_{jk} ln (1- P_{w_j}(\theta_k))



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


