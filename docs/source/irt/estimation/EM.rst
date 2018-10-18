=================
EM算法
=================


EM的一般过程
========================

EM算法的步骤是

- 首先，第一轮迭代t=0时，随机初始化待求参数 :math:`\pmb{\omega}^{(t)}`
- E步，求解隐藏变量 :math:`\pmb{\theta}` 的后验概率密度函数 :math:`g(\pmb{\theta} | \mathbf{Y},\pmb{\omega}^{(t)} )`
    - :math:`\pmb{\theta}` 本身是连续值，这时  :math:`g(\pmb{\theta} | \mathbf{Y},\pmb{\omega}^{(t)} )` 就是概率密度函数，计算积分比较复杂。
    - 所以可以把 :math:`\pmb{\theta}` 离散化，这样 :math:`g(\pmb{\theta} | \mathbf{Y},\pmb{\omega}^{(t)} )` 就是概率质量函数，只需要求出其概率分布，然后利用求和的方式计算全概率。
- M步，极大化Q函数 :math:`Q(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)})` 得到新的  :math:`\pmb{\omega}^{(t+1)}`

.. math::
    \pmb{\omega}^{(t+1)} = \mathop{\arg\max}_{\pmb{\omega} \in \pmb{\Omega}}
    \sum_{\pmb{\theta} \in \pmb{\Theta}} \left [ g(\pmb{\theta}|\mathbf{Y},\pmb{\omega}^{(t)}) ln L(\mathbf{Y},\pmb{\theta}|\pmb{\omega}) \right ]


- 重复E步和M步直到满足收敛条件
    - :math:`\pmb{\omega}` 不再变化 :math:`|\pmb{\omega}^{(t+1)} - \pmb{\omega}^{(t)}|<\epsilon`
    - 对数似然函数不再变化 :math:`|lnL(\mathbf{Y}|\pmb{\omega}^{(t+1)}) - lnL(\mathbf{Y}|\pmb{\omega}^{(t)})|<\epsilon`




在IRT模型中，我们把学生能力值 :math:`\pmb{\theta}` 看做是隐变量，把题目参数看做未知参数 :math:`\pmb{\omega}` ，双参数的IRT模型公式为：

.. math::
    P(\theta,a,b) = \frac{1}{1+e^{-Da(\theta-b)}}

我们设 :math:`\omega_j=[a_j,b_j]^T` 为题目参数，j代表第j个题目；:math:`\theta_i` 为学生潜在能力值，i代表第i个学生。
矩阵 :math:`\mathbf{Y}=[Y_{ij}]` 为作答记录数据，也就是观测数据。


现在的问题就是，我们如何求得  :math:`\pmb{\theta}` 的后验概率，以及对数似然函数 :math:`ln L(\mathbf{Y},\pmb{\theta}|\pmb{\omega})` 。




目标函数
========================


首先说明一些假设

**假设1，学生的作答行为是相互独立事件**，在作答题目时，学生与学生之间互不影响。

作答记录 :math:`\mathbf{Y}_r` 和  :math:`\mathbf{Y}_s` 分别是学生r、学生s的作答记录向量，
那么， :math:`L(\mathbf{Y}_r,\mathbf{Y}_s|\pmb{\omega})=L(\mathbf{Y}_r|\pmb{\omega})L(\mathbf{Y}_s|\pmb{\omega})` 。
因此，对数似然函数可以改写成

.. math::
    ln L(\mathbf{Y}|\pmb{\omega}) = ln \prod_{i=1}^N L(\mathbf{Y}_i|\pmb{\omega}) = \sum_{i=1}^N ln L(\mathbf{Y}_i|\pmb{\omega})
    \tag{11}

其中i=1,2,3,...,N,i代表学生编号, :math:`\mathbf{Y}_i` 代表学生i作答 :math:`\mathbf{J}` 个题目的作答向量，题目的参数向量是 :math:`\pmb{\omega}`





**假设2，题目的作答是相互独立事件** ，同一个学生的作答记录中，题目与题目之间互相不影响。

.. math::
    L(\mathbf{Y}_i,\pmb{\theta}|\pmb{\omega})=L(\mathbf{Y}_i|\pmb{\theta},\pmb{\omega}) p(\pmb{\theta}|\pmb{\omega})
    =\prod_{j=1}^J L(\mathbf{Y}_{ij}|\pmb{\theta},\pmb{\omega}_j) p(\pmb{\theta}|\pmb{\omega}_j)
    \tag{13}

公式(13)表达的是一个学生的J个题目对数似然函数。

**假设3，题目参数和学生能力值是相互独立的** ， :math:`p(\pmb{\theta},\pmb{\omega})=p(\pmb{\theta})p(\pmb{\omega})` ,
并且，:math:`p(\pmb{\theta}|\pmb{\omega})=p(\pmb{\theta})`
这里 :math:`p(\pmb{\theta})` 和 :math:`p(\pmb{\omega})` 是先验概率

.. math::
      L(\mathbf{Y}_i,\pmb{\theta}| \pmb{\omega}) &=  \prod_{j=1}^J l(Y_{ij}|\pmb{\theta},\pmb{\omega}_j) p(\pmb{\theta} | \pmb{\omega}_j) \\
      &=  \prod_{j=1}^J l(Y_{ij}|\pmb{\theta},\pmb{\omega}_j) p(\pmb{\theta}) \\


**假设4，学生能力值是单维变量**

好像是废话，IRT是分为单维能力模型和多维能力模型的，本篇讲的的单维IRT模型，也就是对于一个学生来说，其能力值 :math:`\theta` 是一个标量值。
事实上，EM算法也只能解决单维IRT模型的参数估计。





将似然函数代入到Q函数，其中 :math:`ln\ p(\theta_k)` 是theta的先验部分，其值的固定值，在极大化Q过程中不影响，所以可以忽略。

.. math::

    Q(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)}) &= \sum_{\pmb{\theta} \in \pmb{\Theta}} \left [ g(\pmb{\theta}|\mathbf{Y}_i,\pmb{\omega}^{(t)}) ln L(\mathbf{Y},\pmb{\theta}|\pmb{\omega}^{t+1}) \right ] \\
    &= \sum_{k=1}^K \left [ g(\theta_k |\mathbf{Y}_i,\pmb{\omega}^{(t)}) ln L(\mathbf{Y},\theta_k|\pmb{\omega}^{t+1}) \right ] \\
    &= \sum_{i=1}^N \sum_{k=1}^K \left [ g(\theta_k |\mathbf{Y}_i,\pmb{\omega}^{(t)}) ln  L(\mathbf{Y}_i,\theta_k|\pmb{\omega}^{t+1}) \right ] \\
    &= \sum_{i=1}^N \sum_{k=1}^K \left [ g(\theta_k |\mathbf{Y}_i,\pmb{\omega}^{(t)}) \left[ \sum_{j=1}^J ln  L(Y_{ij}|\theta_k,\pmb{\omega}_j^{t+1}) +ln p(\theta_k)  \right ] \right ] \\
    \\
    &\Leftrightarrow   \sum_{i=1}^N \sum_{j=1}^J \sum_{k=1}^K
    \left [ g(\theta_k|\mathbf{Y}_i,\pmb{\omega}^{(t)})
    ln\ l(Y_{ij}|\theta_k,\pmb{\omega}_j^{t+1})  \right ]


项目反映理论模型和二项式非常类似，学生i作答题目j的结果变量 :math:`Y_{ij}` 概率分布函数为

.. math::
    l(Y_{ij}|\theta_k,\pmb{\omega}_j) =  P_{\omega_j}(\theta_k) ^{Y_{ij}} (1- P_{\omega_j}(\theta_k) )^{1-Y_{ij}}

其中，:math:`P_{\omega_j}(\theta_k)=P(\theta,a,b) = \frac{1}{1+e^{-Da(\theta-b)}}` .

.. math::
    ln L(\mathbf{Y}_{ij}=x_{ij}|\theta_k,\pmb{\omega}_j) &=
    ln \left [  P_{\omega_j}(\theta_k)^{Y_{ij}} \left(1-P_{\omega_j}(\theta_k)\right)^{1-Y_{ij}}  \right ] \\
    &= Y_{ij} ln P_{\omega_j}(\theta_k) + (1-Y_{ij})ln \left(1-P_{\omega_j}(\theta_k)\right)



在EM算法的M步，每一轮迭代极大化Q函数，求得题目参数 :math:`\pmb{\omega}`

.. math::
    \pmb{\omega}^{(t+1)} &=  \mathop{\arg\max}_{\pmb{\omega} \in \Omega}
    \sum_{i=1}^N \sum_{j=1}^J \sum_{k=1}^K
        \left [ g(\theta_k|\mathbf{Y}_i,\pmb{\omega}^{(t)})
        ln\ l(Y_{ij}|\theta_k,\pmb{\omega}_j^{t+1})  \right ] \\
    &= \mathop{\arg\max}_{\pmb{\omega} \in \Omega}
    \sum_{j=1}^J \sum_{i=1}^N \sum_{k=1}^K g(\theta_k|\mathbf{Y}_i,\pmb{\omega}^{(t)})
    \left[   Y_{ij} ln P_{\omega_j}(\theta_k) + (1-Y_{ij})ln \left(1-P_{\omega_j}(\theta_k)\right) \right]



其中可以进一步化简,固定的j，我们令

.. math::
    \hat{r}_{jk} = \sum_{i=1}^N g(\theta_k|\mathbf{Y}_i,\pmb{\omega}^{(t)}) Y_{ij}

.. math::
     \hat{W}_{jk} = \sum_{i=1}^N g(\theta_k|\mathbf{Y}_i,\pmb{\omega}^{(t)}) (1-Y_{ij})

最终结果：

.. math::

    \pmb{\omega}^{(t+1)} = \mathop{\arg\max}_{\pmb{\omega} \in \pmb{\Omega}} \sum_{j=1}^J \sum_{k=1}^{K} \hat{r}_{jk} ln P_{w_j}(\theta_k) + \hat{W}_{jk} ln (1- P_{w_j}(\theta_k))



先验概率分布
========================

前文多次提到，随机变量 :math:`\theta` 是连续值，需要进行积分，不利于我们计算，所以我们可以把它进行离散化，方便我们计算其概率分布。
假设 :math:`\theta` 的取值空间是 :math:`\Theta` ,空间 :math:`\Theta` 是实数值空间，
我们从空间 :math:`\Theta` 取K个值，作为 :math:`\theta` 的取值，也就是我们把 :math:`\theta` 从无限的取值空间强制变成只能取K个值。
:math:`\theta_k` 表示 :math:`\theta` 的第k个取值。

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





后验概率分布
========================

其中，能力值 :math:`\theta` 的后验概率分布，求后验概率分布时， :math:`\pmb{\omega}` 的值认为是已知的，
用的上一轮的估计值。 至于 :math:`\pmb{\theta}` 自然是只能取先验中固定的值。

在实际计算的过程中，公式中的 :math:`\prod_{j=1}^J L(Y_{ij}|\theta_k,\pmb{\omega}_j)` 部分存在连乘，
所以会先加一个对数操作，算完后再用指数计算得到正确值。



.. math::

    g(\theta_k|\mathbf{Y}_i,\pmb{\omega}) = \frac{\prod_{j=1}^J L(Y_{ij}|\theta_k,\pmb{\omega}_j) p(\theta_k)}
    {\sum_{m=1}^K \prod_{j=1}^J L(Y_{ij}|\theta_m,\pmb{\omega}_j) p(\theta_m)}



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






极大化目标函数
========================


.. math::

    \pmb{\omega}^{(t+1)} = \mathop{\arg\max}_{\pmb{\omega} \in \pmb{\Omega}} \sum_{j=1}^J \sum_{k=1}^{K} \hat{r}_{jk} ln P_{w_j}(\theta_k) + \hat{W}_{jk} ln (1- P_{w_j}(\theta_k))

一般我们是通过迭代的方式极大化函数求解，需要参数的梯度（偏导）。

在极大化的过程中，**每道题目是可以分开独立估计的** ，所以我们一次只处理一个题目。

我们令 :math:`\hat{y}_{jk} = P_{w_j}(\theta_k)`

一阶导数（Jacobian矩阵）
---------------------------
单独一道题目j的一阶偏导为：

.. math::

    \frac{\partial Q_j} {\partial a_j}  &= \sum_{k=1}^K ( \hat{r}_{jk} - (\hat{r}_{jk}+\hat{W}_{jk})\hat{y}_{jk}) \theta_k

    \frac{\partial Q_j} {\partial b_j}  &= \sum_{k=1}^K ( \hat{r}_{jk} - (\hat{r}_{jk}+\hat{W}_{jk})\hat{y}_{jk})



二阶导数（Hessian矩阵）
---------------------------

.. math::

    \frac{\partial^2 Q_j} {\partial a_j^2}  = - \sum_{k=1}^K (\hat{r}_{jk}+\hat{W}_{jk}) \hat{y}_{jk} ( 1 - \hat{y}_{jk}) \theta_k^2

    \frac{\partial^2 Q_j }{\partial b_j^2}  = - \sum_{k=1}^K (\hat{r}_{jk}+\hat{W}_{jk}) \hat{y}_{jk} ( 1 - \hat{y}_{jk})

     \frac{\partial^2 Q_j }{\partial a_jb_j}  = - \sum_{k=1}^K (\hat{r}_{jk}+\hat{W}_{jk}) \hat{y}_{jk} ( 1 - \hat{y}_{jk})\theta_k




.. math::

    H(a,b|Y,\theta)=\left\{
    \begin{aligned}
    \frac{\partial^2 Q_j} {\partial a_j^2},\ \frac{\partial^2 Q_j }{\partial a_jb_j}

    \frac{\partial^2 Q_j }{\partial a_jb_j} ,\ \frac{\partial^2 Q_j }{\partial b_j^2}
    \end{aligned}
    \right\}.





参考内容
===============================================
    [1] `IRT Parameter Estimation using the EM Algorithm <http://www.openirt.com/b-a-h/papers/note9801.pdf>`_

    [2] `RoutledgeHandbooks-9781315736013-chapter3 <https://www.routledgehandbooks.com/doi/10.4324/9781315736013.ch3>`_

    [3] `Optimizing Information Using the Expectation-Maximization Algorithm in Item Response Theory <https://www.lsac.org/docs/default-source/research-(lsac-resources)/rr-11-01.pdf?sfvrsn=2>`_

    [4] Modern Approaches to Parameter Estimation in Item Response Theory
