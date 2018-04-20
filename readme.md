
# pymc3的问题
## 内存问题

参数太多或者抽样太多都会导致内存问题。

内存出题的情景有二：
1. 多进程时（njobs>1）， 进程间通信传输数据过大
2. 单进程时（njobs>1），太大也会超过内存限制

解决方法
1. pymc3.sample(discard_tuned_samples=False),必须设置为discard_tuned_samples=False
2. 修改pymc3/sampling.py:664

```python
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
                 
                 
```

## 性能问题

pymc3效率比较差，而且还伴随内存问题。主要导致原因是其每次抽样的样本都要记录下来，这样有两个不良影响：
1. 效率差，每次记录都要消耗时间
2. 占用内存大，在变量较多时，记录太多消耗内存巨大。

解决方法：

自己实现一个backends，抛弃前期burn-in的样本，有效样本只保留累计值和数量，
最后求个平均值就行了，毕竟一般我们也只是要抽样的平均值（期望）


# 贝叶斯知识追踪

C++ 版本的BKT
https://github.com/myudelson/hmm-scalable

实验数据
http://pslcdatashop.web.cmu.edu/KDDCup/

国际教育领域数据挖掘协会

http://educationaldatamining.org/resources/
https://github.com/IEDMS

卡内基梅隆大学的贝叶斯学生模型工具

http://www.cs.cmu.edu/~listen/BNT-SM/