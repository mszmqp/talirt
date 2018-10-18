==============
最大后验估计
==============
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