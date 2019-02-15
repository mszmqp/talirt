==========================================
some important
==========================================



开源数据集
==========================================


kdd cup 2010 学生作答数据

http://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp

http://neuron.csie.ntust.edu.tw/homework/98/NN/KDDCUP2010/Dataset/


国际协会(组织)
==========================================

国际教育领域数据挖掘协会

http://educationaldatamining.org/resources/

https://github.com/IEDMS

卡内基梅隆大学的贝叶斯学生模型工具

http://www.cs.cmu.edu/~listen/BNT-SM/


pymc3的问题
==========================================

内存问题
------------------


参数太多或者抽样太多都会导致内存问题。

内存出题的情景有二：
1. 多进程时（njobs>1）， 进程间通信传输数据过大
2. 单进程时（njobs>1），太大也会超过内存限制

解决方法
1. pymc3.sample(discard_tuned_samples=False),必须设置为discard_tuned_samples=False
2. 修改pymc3/sampling.py:664

.. code-block:: python

    def _iter_sample(draws, step, start=None, trace=None, chain=0, tune=None,
                     model=None, random_seed=None):
                     ...

        try:
            step.tune = bool(tune)
            for i in range(draws):
                if i == tune:
                    step = stop_tuning(step)
                if step.generates_stats:
                    point, states = step.step(point)
                    if strace.supports_sampler_stats:
                        if i>= tune: # 增加行
                            strace.record(point, states)
                    else:
                        if i>=tune: # 增加行
                            strace.record(point)
                else:
                    point = step.step(point)
                    if i>=tune: # 增加行
                        strace.record(point)
                yield strace




性能问题
------------------
pymc3效率比较差，而且还伴随内存问题。主要导致原因是其每次抽样的样本都要记录下来，这样有两个不良影响：
1. 效率差，每次记录都要消耗时间
2. 占用内存大，在变量较多时，记录太多消耗内存巨大。

解决方法：

自己实现一个backends，抛弃前期burn-in的样本，有效样本只保留累计值和数量，
最后求个平均值就行了，毕竟一般我们也只是要抽样的平均值（期望）


缺失值问题
------------------
学生答题有缺失值，对于观测变量缺失值的情况，pymc3是支持缺失值的http://docs.pymc.io/notebooks/getting_started中有一段

::
    Missing values are handled transparently by passing a MaskedArray or a pandas.
    DataFrame with NaN values to the observed argument when creating an observed stochastic random variable.
    Behind the scenes, another random variable, disasters.missing_values is created to model the missing values.
    All we need to do to handle the missing values is ensure we sample this random variable as well.

    Unfortunately because they are discrete variables and thus have no meaningful gradient,
    we cannot use NUTS for sampling switchpoint or the missing disaster observations.
    Instead, we will sample using a Metroplis step method, which implements adaptive Metropolis-Hastings,
    because it is designed to handle discrete values.
    PyMC3 automatically assigns the correct sampling algorithms.


**但是实验发现，这么搞抽样会非常慢**


burn-in 数量
------------------------------------


sampler默认是nuts，经实验 burn-in 1000和10000没区别不大。
