# from . import data
# from . import irt
import math


# from uirt_lib import *

def trunk_split(n, slice):
    # if n <=slice:
    if slice <= 0:
        return
    piece = math.ceil(n / slice)
    start = 0
    end = start + piece
    while start < n:
        yield (start, end)
        start += piece
        end = min(start + piece, n)
        # print(end)


def _test_uirt_lib():
    import uirt_lib
    import numpy as np
    from scipy.special import expit as sigmod
    def uirt_prob(theta, b, a=None, c=None, d=1.702):
        theta = theta.reshape(theta.size, 1)
        b = b.reshape(1, b.size)
        if a is None:
            a = np.ones((1, b.size))
        else:
            a = a.reshape(1, a.size)
        if c is None:
            c = np.zeros((1, b.size))
        else:
            c = c.reshape(1, c.size)
        z = d * a * (theta - b)
        return c + (1 - c) * sigmod(z)

    theta = np.array([-2, -1, 0, 1, 2], dtype=np.float)
    a = np.array([1, 1.3, 1.4], dtype=np.float)
    b = np.array([-1, 0, 1], dtype=np.float)
    c = np.array([0.1, 0.2, 0.3], dtype=np.float)

    res3 = uirt_lib.u3irt_matrix(theta=theta, a=a, b=b, c=c)
    assert np.sum(res3 - uirt_prob(theta, a=a, b=b, c=c)) == 0

    res2 = uirt_lib.u2irt_matrix(theta=theta, a=a, b=b)
    assert np.sum(res2 - uirt_prob(theta, a=a, b=b)) == 0

    res1 = uirt_lib.u1irt_matrix(theta=theta, b=b)
    assert np.sum(res1 - uirt_prob(theta, b=b)) == 0


if __name__ == "__main__":
    _test_uirt_lib()
