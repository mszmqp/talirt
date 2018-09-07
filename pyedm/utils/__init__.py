# from . import data
# from . import irt
import math
import numpy as np
from scipy.special import logsumexp

def cpu_count():
    """Try to guess the number of CPUs in the system.

    We use the number provided by psutil if that is installed.
    If not, we use the number provided by multiprocessing, but assume
    that half of the cpus are only hardware threads and ignore those.
    """
    try:
        import psutil
        cpus = psutil.cpu_count(False)
    except ImportError:
        import multiprocessing
        try:
            cpus = multiprocessing.cpu_count() // 2
        except NotImplementedError:
            cpus = 1
    if cpus is None:
        cpus = 1
    return cpus

def normalize(a, axis=None):
    """Normalizes the input array so that it sums to 1.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    a_sum = a.sum(axis)
    if axis and a.ndim > 1:
        # Make sure we don't divide by zero.
        a_sum[a_sum == 0] = 1
        shape = list(a.shape)
        shape[axis] = 1
        a_sum.shape = shape

    a /= a_sum
    return a

def log_normalize(a, axis=None):
    """Normalizes the input array so that the exponent of the sum is 1.

    Parameters
    ----------
    a : array
        Non-normalized input data.

    axis : int
        Dimension along which normalization is performed.

    Notes
    -----
    Modifies the input **inplace**.
    """
    a_lse = logsumexp(a, axis)
    a -= a_lse[:, np.newaxis]


def iter_from_X_lengths(X, lengths):
    if lengths is None:
        yield 0, len(X)
    else:
        n_samples = X.shape[0]
        end = np.cumsum(lengths).astype(np.int32)
        start = end - lengths
        if end[-1] > n_samples:
            raise ValueError("more than {0:d} samples in lengths array {1!s}"
                             .format(n_samples, lengths))

        for i in range(len(lengths)):
            yield start[i], end[i]


def log_mask_zero(a):
    """Computes the log of input probabilities masking divide by zero in log.
    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.
    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)


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
