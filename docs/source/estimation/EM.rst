=================
EM算法
=================

最大似然
========================

机器学习的一般过程是：

1. 对数据进行建模。比如 线性回归、逻辑回归
2. 定义目标函数。比如，最小二乘法、交叉熵、最大似然。
3. 极大（小）化目标函数，求解参数的值。

最大似然估计是一中普遍用于概率模型中定义目标函数的方法。
假设我们的数据样本集合为 :math:`\pmb{Y}` (不是数据特征，是结果变量，比如二分类中的分类结果)，样本数量为N，
模型参数为 :math:`\pmb{\omega}` ，似然函数为：

.. math::
    object\ function = L(\mathbf{Y}|\pmb{\omega})  = P(\mathbf{Y}|\pmb{\omega}) = \prod_{i=1}^N P(Y_i|\pmb{\omega}) \tag{1}

.. note::
    似然函数的概率意义为，当模型参数为 :math:`\pmb{\omega}` 时，所有样本 :math:`\pmb{Y}` 同时发生的概率，所以是连乘。

上式中有连乘操作，为了避免计算机浮点数溢出，会增加log把乘法变成加法（这对于极大化求解的参数值时等价的）。

.. math::
    object\ function = ln L(\mathbf{Y}|\pmb{\omega})  = ln P(\mathbf{Y}|\pmb{\omega}) = \sum_{i=1}^N ln\ P(Y_i|\pmb{\omega})




我们极大化目标函数，求解出 :math:`\pmb{\omega}`

.. math::

     \pmb{\omega} &= \mathop{\arg\max} \ ln\ L(\mathbf{Y}|\pmb{\omega})  \\
      &= \mathop{\arg\max}\ ln\ \prod_{i=1}^N l(Y_i|\pmb{\omega}) \\
      &= \mathop{\arg\max}\ \sum_{i=1}^N\ ln\ l(Y_i|\pmb{\omega}) \ \ \ \ \text{(2)}

贝叶斯估计
==============================

回顾一下贝叶斯公式：

.. math::
    P(\pmb{\omega}|\mathbf{Y})= \frac{P(\mathbf{Y}|\pmb{\omega}) P(\pmb{\omega})}{\sum_{\pmb{\omega} \in \pmb{\Omega}} P(\mathbf{Y}|\pmb{\omega}) P(\pmb{\omega}) } \\
    \mbox{后验概率（条件概率）} = \frac{\mbox{似然（条件概率）} * \mbox{先验概率}}{\mbox{全概率公式}}



最大后验估计（MAP），就是极大化参数  :math:`\pmb{\omega}` 的后验概率求解出  :math:`\pmb{\omega}` 的值。
上式中的分母部分其实是一个常量，因为分母是对 :math:`\pmb{\omega}` 进行了边际化，也就是无论 :math:`\pmb{\omega}` 取什么值都不会影响分母最终的结果。
其实分母项也可以认为是规则化项，就是为了使得最后后验概率分布满足 :math:`\sum_{\pmb{\omega} \in \pmb{\Omega}} = 1` 。

所以极大化后验概率只需要极大化分子部分即可。

.. math::
    P(\pmb{\omega}|\mathbf{Y}) \propto  P(\mathbf{Y}|\pmb{\omega}) P(\pmb{\omega})

这样我们的目标函数就成了

.. math::
    object \ function = \underbrace { P(\mathbf{Y}|\pmb{\omega})}_{\text{似然函数}} \underbrace {P(\pmb{\omega}) }_{参数的先验概率}


很显然，**最大后验估计就是在最大似然估计的基础上乘以参数的先验概率，当数据样本数N无穷大时，最大后验估计其实是等同于最大似然估计的值** 。

增加对数函数

.. math::
    object \ function = \underbrace {ln P(\mathbf{Y}|\pmb{\omega})}_{\text{对数似然}} + \underbrace {ln\ P(\pmb{\omega})}_{\text{对数先验}}



.. note::
    增加的先验部分，也可以理解成是为了防止过拟合的正则项。


我们极大化目标函数，求解出 :math:`\pmb{\omega}`

.. math::

     \pmb{\omega} &= \mathop{\arg\max} \ ln\ L(\mathbf{Y}|\pmb{\omega})  + ln\ P(\pmb{\omega}) \\
      &= \mathop{\arg\max} \ ln\ \prod_{i=1}^N l(Y_i|\pmb{\omega}) + ln\ P(\pmb{\omega}) \\
      &= \mathop{\arg\max} \sum_{i=1}^N ln\ l(Y_i|\pmb{\omega})  + ln\ P(\pmb{\omega}) \ \ \ \ \text{(3)}

EM原理
========================

上文中，我们回顾了最大似然估计和最大后验概率估计，两者很相似，都是先定义出目标函数，然后极大化目标函数求解，
二者唯一的区别就最大后验估计的目标函数相比于最大似然估计的目标函数多了一个参数的先验函数。

有些时候我们的模型中，除了参数  :math:`\pmb{\omega}` 外，还会包含另外的变量，而这个变量的值是未知的，我们称之为 **隐变量** 。

从上文中，我们知道目标函数是关于参数  :math:`\pmb{\omega}` 的一个函数  :math:`object\ function = f(\pmb{\omega}|\mathbf{Y})` ,
如果模型中增加了额外的未知参数  :math:`\pmb{\theta}` ，目标函数就可以改写成：

.. math::
    object\ function = f(\pmb{\omega},\pmb{\theta}|\mathbf{Y})




比如在IRT模型中，除了项目参数  :math:`\pmb{\omega}` 外， 学生能力参数 :math:`\pmb{\theta}` 也是未知的，这时学生能力参数可以看做是隐变量。

根据边际化理论，我们可以通过对隐变量参数 :math:`\pmb{\theta}` 进行边际化，以消除它的影响。


上述 **目标函数需要转换成边际似然** 。

.. note::
    当不知道一个随机变量的取值时，对一个随机变量进行边际化（边缘化），就相当消去这个随机变量的影响。

.. math::


     object\ function = lnL(\mathbf{Y}|\pmb{\omega}) = ln \sum_{\theta \in \Theta} L(\mathbf{Y},\pmb{\theta}|\pmb{\omega}) \tag{3}

.. note::
    回顾前文的最大似然估计，
    上式中的函数 :math:`L(\mathbf{Y},\pmb{\theta}|\pmb{\omega})` 是似然函数。

.. note::
    在边际化的时候，对于连续值随机变量应该是积分操作，对于离散值随机变量是求和。这里为了简化公式，我们假设 :math:`\pmb{\theta}` 是离散随机变量，这样就可以用求和的方式进行边际化。



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

    lnL(\mathbf{Y}|\pmb{\omega}) &= \ ln  \sum_{\theta \in \Theta} L(\mathbf{Y},\pmb{\theta}|\pmb{\omega})  \\
    &= ln \sum_{\theta \in \Theta} g(\pmb{\theta}) \frac{L(\mathbf{Y},\pmb{\theta}|\pmb{\omega})}{g(\pmb{\theta})}  \\
    &\ge \sum_{\theta \in \Theta} g(\pmb{\theta}) ln \frac{L(\mathbf{Y},\pmb{\theta}|\pmb{\omega})}{g(\pmb{\theta})}




上式中， :math:`g(\pmb{\theta})` 是 :math:`\pmb{\theta}` 的概率密度（质量）函数，公式第2行，乘以一个 :math:`g(\pmb{\theta})` ，
然后再除以一个 :math:`g(\pmb{\theta})` 等式不变。 :math:`\sum_{\theta \in \Theta} g(\pmb{\theta})` 相当于对后面的
部分 :math:`\frac{L(\mathbf{Y},\pmb{\theta}|\pmb{\omega})}{g(\pmb{\theta})}` 求期望，这样结合Jensen不等式得出第三行。

公式(5)提供了似然函数的一个下边界，极大化下边界函数，和极大化似然函数可以得到相似的结果。 :math:`g(\pmb{\theta})` 我们定义成 :math:`\pmb{\theta}` 的
后验概率密度函数  :math:`g(\pmb{\theta} | \mathbf{Y},\pmb{\omega})` , 公式(5)就可以转换成

.. math::
    lnL(\mathbf{Y}|\pmb{\omega}) = \ ln  \sum_{\pmb{\theta} \in \Theta} L(\mathbf{Y},\pmb{\theta}|\pmb{\omega})
    \Leftrightarrow  \sum_{\pmb{\theta} \in \Theta} g(\pmb{\theta} | \mathbf{Y},\pmb{\omega})
    ln \frac{L(\mathbf{Y},\pmb{\theta}|\pmb{\omega})}{g(\pmb{\theta}|\mathbf{Y},\pmb{\omega})}
    \tag{6}


**通过Jensen不等式的转换， 我们把 对数ln内部的求和（积分）操作转化到外部，简化了计算。**

.. note::
    :math:`\pmb{\theta}` 是学生能力值，我们看作一个随机变量。既然是随机变量就会有其概率分布。
    对于连续值随机变量，其概率分布函数称为概率密度函数， **函数值不是概率值** ，需要进行积分才能得出概率值（不懂的，去恶补概率论基础）；
    对于离散值随机变量，其概率分布函数称为概率质量函数， **函数值就是概率值** 。
    公式中的 :math:`g(\pmb{\theta} | \mathbf{Y},\pmb{\omega})` 就是随机变量 :math:`\pmb{\theta}` 的概率分布函数。
    当然，这里 :math:`\pmb{\theta}` 学生能力值应该是连续值，但是积分操作过于复杂，所以我们可以把 :math:`\pmb{\theta}` 进行离散化，
    把 :math:`\pmb{\theta}` 转换成离散值，这样就不用积分操作了。离散化的方法以及 :math:`g(\pmb{\theta} | \mathbf{Y},\pmb{\omega})` 如何求下文介绍。







依据对数的性质，公式(6)中的除法可以变成减法

.. math::
    lnL(\mathbf{Y}|\pmb{\omega}) =
    \underbrace {\sum_{\pmb{\theta} \in \Theta} g(\pmb{\theta}|\mathbf{Y},\pmb{\omega}) ln L(\mathbf{Y},\pmb{\theta}|\pmb{\omega})}_{Q(\omega|\omega^{(t)})}
    - \underbrace {\sum_{\pmb{\theta} \in \Theta} g(\pmb{\theta}|\mathbf{Y},\pmb{\omega}) ln \ g(\pmb{\theta}|\mathbf{Y},\pmb{\omega})}_{H(\omega|\omega^{(t)})}
    \tag{8}


极大化目标函数是一个迭代的过程，我们令t表示迭代的序号，t=0,1,2,...。公式(8)中包含两部分， :math:`Q(\pmb{\omega}|\pmb{\omega}^{(t)})`
和 :math:`H(\pmb{\omega}|\pmb{\omega}^{(t)})` ，
其中 :math:`Q(\pmb{\omega}|\pmb{\omega}^{(t)})` 表示 **完全数据的似然函数** ，也就李航<统计学习>中讲的Q函数。同时，也是在EM算法的M步中要极大化的目标函数。

:math:`H(\pmb{\omega}|\pmb{\omega}^{(t)})` 表示潜在能力变量的后验概率密度（其实是关于后验概率密度函数的一个函数）。
根据Jensen不等式，对于任意 :math:`\pmb{\omega} \in \pmb{\Omega}` ,
:math:`H(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)}) \le H(\pmb{\omega}^{(t)}|\pmb{\omega}^{(t)})` ，**所以在极大化过程中可以忽略H部分** 。

在EM算法的M步骤中，每次迭代 :math:`t \rightarrow t+1` 都是最大化 :math:`Q(\pmb{\omega}|\pmb{\omega}^{(t)})` ，
所以可以确保 :math:`Q(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)}) \ge Q(\pmb{\omega}^{(t)}|\pmb{\omega}^{(t)})` 。

这样，随着每一次迭代，一定能确保对数似然函数的值（我们的目标是极大化对数似然函数）是增长的

.. math::
    &lnL(\mathbf{Y}|\pmb{\omega}^{(t+1)}) - lnL(\mathbf{Y}|\pmb{\omega}^{(t)}) \\
    &= [Q(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)}) - H(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)})]
        - [ Q(\pmb{\omega}^{(t)}|\pmb{\omega}^{(t)}) - H(\pmb{\omega}^{(t)}|\pmb{\omega}^{(t)})] \\
    &= \underbrace{[Q(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)})- Q(\pmb{\omega}^{(t)}|\pmb{\omega}^{(t)})]}_{\ge0} +
        \underbrace{[H(\pmb{\omega}^{(t)}|\pmb{\omega}^{(t)}) - H(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)})]}_{\ge0} \\
    &\ge 0





完整的推导过程是:


.. math::
    ln L(\mathbf{Y}|\pmb{\omega})
    &= ln \sum_{\theta \in \Theta} L(\mathbf{Y},\pmb{\theta}|\pmb{\omega}) \\
    &\Leftrightarrow  \sum_{\theta \in \Theta} g(\pmb{\theta}|\mathbf{Y},\pmb{\omega})
        ln \frac{L(\mathbf{Y},\pmb{\theta}|\pmb{\omega})}{g(\pmb{\theta}|\mathbf{Y},\pmb{\omega})} \ \ \text{根据Jensen不等式}\\
    &= \underbrace{\sum_{\theta \in \Theta} g(\pmb{\theta}|\mathbf{Y},\pmb{\omega})\ ln L(\mathbf{Y},\pmb{\theta}|\pmb{\omega}) }_Q
    - \underbrace{ \sum_{\theta \in \Theta} g(\pmb{\theta}|\mathbf{Y},\pmb{\omega}) g(\pmb{\theta}|\mathbf{Y},\pmb{\omega})}_H
    \\
    &\Leftrightarrow \underbrace{\sum_{\theta \in \Theta} g(\pmb{\theta}|\mathbf{Y},\pmb{\omega})\ ln L(\mathbf{Y},\pmb{\theta}|\pmb{\omega}) }_Q
    \ \ \text{H部分可忽略}

根据前文的推导，我们需要极大化Q函数部分，求得参数 :math:`\pmb{\omega}` ,

.. math::
    object\ function = Q =  \sum_{\theta \in \Theta} g(\pmb{\theta}|\mathbf{Y},\pmb{\omega})\ ln L(\mathbf{Y},\pmb{\theta}|\pmb{\omega})



现在我们laikan


但是Q函数中包含三部分。

.. math::

    Q=\sum_{i=1}^N \sum_{k=1}^K \left [ \underbrace{g(\theta_k|\mathbf{Y}_i,\pmb{\omega})}_{\text{能力值后验概率}}
         \left [ \sum_{j=1}^J \underbrace{ln\ L(Y_{ij}|\theta_k,\pmb{\omega}_j)}_{\text{对数似然函数}}+
          \underbrace{ ln \ p(\theta_k)}_{\text{能力值对数先验分布}} \right ] \right ]  \tag{10}

在EM算法的M步，每一轮迭代极大化Q函数，求得题目参数 :math:`\pmb{\omega}` 。
能力值对数先验分布部分与参数 :math:`\pmb{\omega}` 无关，在极大化Q的过程中值的固定不变的，
不影响我们对参数 :math:`\pmb{\omega}` 的求解，所以是可以忽略的 。


.. math::
    \pmb{\omega}^{(t+1)} &= \mathop{\arg\max}_{\pmb{\omega} \in \Omega} Q \\
    &=  \mathop{\arg\max}_{\pmb{\omega} \in \Omega} \sum_{i=1}^N \sum_{k=1}^K g(\theta_k|\mathbf{Y}_i,\pmb{\omega}^{(t)})
     \sum_{j=1}^J  ln\ L(Y_{ij}|\theta_k,\pmb{\omega}_j)
    \qquad \qquad \text{(11)}





隐变量的后验分布
======================

随机变量 :math:`\pmb{\theta}` 概率分布函数是什么呢？ 我们既然已经有了学生的作答记录，
那么，我们可以利用学生的作答记录来计算 :math:`\pmb{\theta}` 的后验概率分布。所以公式(6)中使用的是 :math:`\pmb{\theta}` 的后验概率。
根据贝叶斯定理，:math:`\pmb{\theta}` 的后验概率密度函数可以定义为：

.. math::
     g(\pmb{\theta} | \mathbf{Y},\pmb{\omega} ) =
    \frac{L( \mathbf{Y}|\pmb{\theta},\pmb{\omega})p(\pmb{\theta},\pmb{\omega})}
    {\sum_{\pmb{\theta}' \in \Theta} L(\mathbf{Y}|\pmb{\theta}',\pmb{\omega})p(\pmb{\theta}',\pmb{\omega})  }  \tag{7}

.. note::
    什么是先验概率、后验概率，这里不介绍了，自己恶补概率论基础吧。
    根据贝叶斯定理，公式(7)的分母部分应该是全概率公式，如果 :math:`\pmb{\theta}` 是连续值，这里是要对 :math:`\pmb{\theta}` 进行积分的。
    之前我们说过，我们会对 :math:`\pmb{\theta}`  进行离散化，这样就可以把积分操作转换成求和操作。



算法总结
------------------



以上是简单的推导过程，EM算法的步骤是

- 首先，第一轮迭代t=0时，随机初始化待求参数 :math:`\pmb{\omega}^{(t)}`
- E步，求解隐藏变量 :math:`\pmb{\theta}` 的后验概率密度函数 :math:`g(\pmb{\theta} | \mathbf{Y},\pmb{\omega}^{(t)} )`
    - :math:`\pmb{\theta}` 本身是连续值，这时  :math:`g(\pmb{\theta} | \mathbf{Y},\pmb{\omega}^{(t)} )` 就是概率密度函数，计算积分比较复杂。
    - 所以可以把 :math:`\pmb{\theta}` 离散化，这样 :math:`g(\pmb{\theta} | \mathbf{Y},\pmb{\omega}^{(t)} )` 就是概率质量函数，只需要求出其概率分布，然后利用求和的方式计算全概率。
- M步，极大化Q函数 :math:`Q(\pmb{\omega}^{(t+1)}|\pmb{\omega}^{(t)})` 得到新的  :math:`\pmb{\omega}^{(t+1)}`

.. math::
    \pmb{\omega}^{(t+1)} = \mathop{\arg\max}_{\pmb{\omega} \in \pmb{\Omega}}
    \sum_{\pmb{\theta} \in \pmb{\Theta}} g(\pmb{\theta}|\mathbf{Y},\pmb{\omega}^{(t)}) ln L(\mathbf{Y},\pmb{\theta}|\pmb{\omega})
    \tag{10}

- 重复E步和M步直到满足收敛条件
    - :math:`\pmb{\omega}` 不再变化 :math:`|\pmb{\omega}^{(t+1)} - \pmb{\omega}^{(t)}|<\epsilon`
    - 对数似然函数不再变化 :math:`|lnL(\mathbf{Y}|\pmb{\omega}^{(t+1)}) - lnL(\mathbf{Y}|\pmb{\omega}^{(t)})|<\epsilon`


仔细观察M步中要极大化的目标函数Q函数，其实就是完全数据似然函数乘以隐变量的后验概率函数，并且对隐变量的后验概率分布进行边际化。



.. math::
    \pmb{\omega}^{(t+1)} = \mathop{\arg\max}_{\pmb{\omega} \in \pmb{\Omega}}
    \underbrace{ \sum_{\pmb{\theta} \in \pmb{\Theta}} g(\pmb{\theta}|\mathbf{Y},\pmb{\omega}^{(t)}) }_{\text{边际化后验概率}}
    \underbrace{ ln L(\mathbf{Y},\pmb{\theta}|\pmb{\omega})}_{\text{完全数据似然函数}}



当似然函数中含有隐藏变量时（其值未知），就通过边际化的方法去掉隐变量。要边际化一个变量，就需要知道这个变量的概率分布函数，
这里就采用隐变量的后验概率分布作为其概率分布。


.. math::

    &= \sum_{k=1}^K g(\theta_k|\mathbf{Y},\pmb{\omega})\ ln \prod_{i=1}^N l(Y_i,\theta_k|\pmb{\omega})  \\
    &= \sum_{i=1}^N \sum_{k=1}^K g(\theta_k|\mathbf{Y}_i,\pmb{\omega})
        \left [ ln\ l(Y_{i}|\theta_k,\pmb{\omega})  \right ] \\



参考内容
===============================================
　　[1] `IRT Parameter Estimation using the EM Algorithm <http://www.openirt.com/b-a-h/papers/note9801.pdf>`_

　　[2] `RoutledgeHandbooks-9781315736013-chapter3 <https://www.routledgehandbooks.com/doi/10.4324/9781315736013.ch3>`_

　　[3] `Optimizing Information Using the Expectation-Maximization Algorithm in Item Response Theory <https://www.lsac.org/docs/default-source/research-(lsac-resources)/rr-11-01.pdf?sfvrsn=2>`_


