=================
简介
=================



三参数logistic模型：

.. math::
        P(\theta) = c + \frac{(1-c)}{1+e^{-Da(\theta-b)}}


若c=0，简化为双参数logistic模型

.. math::
        P(\theta) = \frac{1}{1+e^{-Da(\theta-b)}}

若c=0其a=1，简化为单参数logistic模型

.. math::
        P(\theta) = \frac{1}{1+e^{-D(\theta-b)}}

Rasch模型

.. math::
        P(\theta) = \frac{e^{\theta-b}}{1+e^{\theta-b}}
