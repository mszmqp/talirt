关于pymc3 memory error 的问题

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
++                  if i>= tune:
                        strace.record(point, states)
                else:
++                  if i>=tune:
                        strace.record(point)
            else:
                point = step.step(point)
++              if i>=tune:
                    strace.record(point)
            yield strace                 
                 
                 
```


