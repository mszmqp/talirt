# cython:profile=False,boundscheck=False, wraparound=False
cimport cython
from cython cimport view
import numpy as np
cimport numpy as np
from numpy.math cimport  logl, log1pl, isinf, fabsl, INFINITY,expl
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdlib cimport malloc, free
# from __future__ import print_function

# from  cimport
# from libcpp.vector cimport vector
# from libcpp.string cimport string
# from libc.stdio cimport printf
# from scipy.misc import logsumexp


ctypedef double dtype_t
cdef enum:

   N_STATS = 2 # 隐状态数量
   N_OBS = 2 # 观测状态数量


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int _argmax(dtype_t[:] X) nogil:
    cdef dtype_t X_max = -INFINITY
    cdef int pos = 0
    cdef int i
    for i in range(X.shape[0]):
        if X[i] > X_max:
            X_max = X[i]
            pos = i
    return pos

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline dtype_t _max(dtype_t[:] X) nogil:
    return X[_argmax(X)]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline dtype_t _max2D(dtype_t[:,:] X) nogil:
    cdef dtype_t X_max = -INFINITY
    cdef int pos = 0
    cdef int i,j
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if X[i,j] > X_max:
                X_max = X[i,j]

    return X_max

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline dtype_t _logsumexp(dtype_t[:] X) nogil:
    cdef dtype_t X_max = _max(X)
    if isinf(X_max):
        return -INFINITY

    cdef dtype_t acc = 0
    cdef int i =0
    for i in range(X.shape[0]):
        acc += expl(X[i] - X_max)

    return logl(acc) + X_max


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline dtype_t _logsumexp_c(dtype_t *X,int length) nogil:
    cdef dtype_t X_max=X[0]
    cdef int i =0
    for i in range(1,length):
        if X[i] > X_max:
            X_max = X[i]

    # cdef dtype_t X_max = _max(X)
    if isinf(X_max):
        return -INFINITY

    cdef dtype_t acc = 0

    for i in range(length):
        acc += expl(X[i] - X_max)

    return logl(acc) + X_max


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline dtype_t _logsumexp2D(dtype_t[:,:] X) nogil:
    cdef dtype_t X_max = _max2D(X)
    if isinf(X_max):
        return -INFINITY

    cdef dtype_t acc = 0
    cdef int i =0
    cdef int j=0
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            acc += expl(X[i,j] - X_max)

    return logl(acc) + X_max

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline dtype_t _logaddexp(dtype_t a, dtype_t b) nogil:
    if isinf(a) and a < 0:
        return b
    elif isinf(b) and b < 0:
        return a
    else:
        return max(a, b) + log1pl(expl(-fabsl(a - b)))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void  _forward(int n_samples, int n_stats,
             dtype_t[:] log_startprob,
             dtype_t[:, :] log_transmat,
              dtype_t[:, :] log_emission,
              int[:] x,
             dtype_t[:, :] fwdlattice):
    """
    
    Parameters
    ----------
    n_samples : int 样本数量
    n_stats : int 状态数量
    log_startprob
    log_transmat
    framelogprob : array-like  shape=(n_samples, n_stats) 每个隐状态到观测值的概率
    fwdlattice : array-like  shape=(n_samples, n_stats) 返回结果

    Returns
    -------

    """
    cdef int t, i, j
    # cdef dtype_t[::view.contiguous] work_buffer = np.zeros(n_stats)
    cdef dtype_t * work_buffer = <dtype_t*> PyMem_Malloc(n_stats * sizeof(dtype_t))
    # cdef dtype_t work_buffer[2]
    with nogil:
        for i in range(n_stats):
            fwdlattice[0, i] = log_startprob[i]  + log_emission[i,x[0]] # framelogprob[0, i]

        for t in range(1, n_samples):
            for j in range(n_stats):
                for i in range(n_stats):
                    work_buffer[i] = fwdlattice[t - 1, i] + log_transmat[i, j]

                # fwdlattice[t, j] = _logsumexp(work_buffer) + log_emission[j,x[t]] # framelogprob[t, j]
                fwdlattice[t, j] = _logsumexp_c(work_buffer,n_stats) + log_emission[j,x[t]] # framelogprob[t, j]

    PyMem_Free(work_buffer)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _backward(int n_samples, int n_stats,
              dtype_t[:] log_startprob,
              dtype_t[:, :] log_transmat,
              dtype_t[:, :] log_emission,
              int[:] x,
              dtype_t[:, :] bwdlattice):

    cdef int t, i, j
    # cdef dtype_t[::view.contiguous] work_buffer = np.zeros(n_stats)
    # cdef dtype_t * work_buffer= new dtype_t[n_stats]
    cdef dtype_t * work_buffer = <dtype_t*> PyMem_Malloc(n_stats * sizeof(dtype_t))
    # cdef dtype_t work_buffer[2]
    with nogil:
        for i in range(n_stats):
            bwdlattice[n_samples - 1, i] = 0.0

        for t in range(n_samples - 2, -1, -1):
            for i in range(n_stats):
                for j in range(n_stats):
                    work_buffer[j] = (log_transmat[i, j]
                                      + log_emission[j,x[t+1]]  #+ framelogprob[t + 1, j]
                                      + bwdlattice[t + 1, j])
                # bwdlattice[t, i] = _logsumexp(work_buffer)
                bwdlattice[t, i] = _logsumexp_c(work_buffer,2)
    PyMem_Free(work_buffer)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef dtype_t** creat2D(int size1, int size2, dtype_t default=0):

    cdef dtype_t ** ar = <dtype_t **>malloc(size1 * sizeof(dtype_t))
    cdef int i=0,j=0
    for i in range(size1):
        ar[i] = <dtype_t *>malloc(size2 * sizeof(dtype_t))
        for j in range(size2):
            ar[i][j] = default
    return ar

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void _compute_xi_sum(int n_samples, int n_stats,
                        dtype_t[:, :] fwdlattice,
                        dtype_t[:, :] log_transmat,
                        dtype_t[:, :] bwdlattice,
                        dtype_t[:, :] log_emission,
                        int[:] x,
                        dtype_t[:, :] xi_sum,double logprob) nogil:

    cdef int t, i, j
    set_constant_2d(xi_sum,n_stats,n_stats,0)


    for t in range(n_samples - 1):
        for i in range(n_stats):
            for j in range(n_stats):
                xi_sum[i][j] += expl(fwdlattice[t, i]
                                     + log_transmat[i, j]
                                     +log_emission[j,x[t+1]]  # + framelogprob[t + 1, j]
                                     + bwdlattice[t + 1, j] - logprob)

        # for i in range(n_stats):
        #     for j in range(n_stats):
        #         log_xi_sum[i, j] = _logaddexp(log_xi_sum[i, j],work_buffer[i][j])
        #         log_xi_sum[i, j] = log_xi_sum[i, j] + work_buffer[i][j]
    # PyMem_Free(work_buffer)



@cython.boundscheck(False)
@cython.wraparound(False)
def _viterbi(int n_samples, int n_stats,
             dtype_t[:] startprob,
             dtype_t[:, :] transmat,
             dtype_t[:, :] emissionprob,
             int[:] obs):
    cdef dtype_t[:] log_startprob
    cdef dtype_t[:, :] log_transmat
    cdef dtype_t[:, :] log_emission
    with np.errstate(divide="ignore") :  # 忽略对0求log的警告
        log_startprob = np.log(startprob)
        log_transmat = np.log(transmat)
        log_emission = np.log(emissionprob)

    cdef int i, j, t, where_from
    cdef dtype_t logprob

    cdef int[::view.contiguous] state_sequence = \
        np.empty(n_samples, dtype=np.int32)
    cdef dtype_t[:, ::view.contiguous] viterbi_lattice = \
        np.zeros((n_samples, n_stats))
    cdef dtype_t[::view.contiguous] work_buffer = np.empty(n_stats)

    with nogil:
        for i in range(n_stats):
            viterbi_lattice[0, i] = log_startprob[i] + log_emission[i,obs[0]] # framelogprob[0, i]

        # Induction
        for t in range(1, n_samples):
            for i in range(n_stats):
                for j in range(n_stats):
                    work_buffer[j] = (log_transmat[j, i]
                                      + viterbi_lattice[t - 1, j])

                viterbi_lattice[t, i] = _max(work_buffer) + log_emission[i,obs[t]] #framelogprob[t, i]

        # Observation traceback
        state_sequence[n_samples - 1] = where_from = \
            _argmax(viterbi_lattice[n_samples - 1])
        logprob = viterbi_lattice[n_samples - 1, where_from]

        for t in range(n_samples - 2, -1, -1):
            for i in range(n_stats):
                work_buffer[i] = (viterbi_lattice[t, i]
                                  + log_transmat[i, where_from])

            state_sequence[t] = where_from = _argmax(work_buffer)

    return np.asarray(state_sequence), expl(logprob)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void reset_zero_1d(dtype_t[:] x ,unsigned int n):
    cdef int i=0
    for i in range(n):
        x[i]=0

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void reset_zero_2d(dtype_t[:,:] x ,unsigned int n_rows,unsigned int n_columns):
    cdef int i=0,j=0
    for i in range(n_rows):
        for j in range(n_columns):
            x[i,j]=0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void set_constant_1d(dtype_t[:] x ,unsigned int n,double value) nogil:
    cdef int i=0
    for i in range(n):
        x[i]= value

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void set_constant_2d(dtype_t[:,:] x ,unsigned int n_rows,unsigned int n_columns,double value) nogil:
    cdef int i=0,j=0
    for i in range(n_rows):
        for j in range(n_columns):
            x[i,j]= value

@cython.boundscheck(False)
@cython.wraparound(False)
def logsumexp(dtype_t[:] X):
    return _logsumexp(X)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fit(int[:] data,
        int[:, :] data_slice,
        dtype_t[:] startprob,dtype_t[:, :] transmat,dtype_t[:, :] emissionprob,
        dtype_t[:] startprob_lb, dtype_t[:] startprob_ub,
        dtype_t[:, :] transmat_lb,dtype_t[:, :] transmat_ub,
        dtype_t[:, :] emissionprob_lb,dtype_t[:, :] emissionprob_ub,
        int max_iter,double tol):
    # 每次循环是一个观测序列
    cdef int n_stats=2
    cdef int n_obs=2,o
    cdef int n_samples=0

    cdef int i,j,k,t,iter
    cdef double log_likelihood=0,cur_log_likelihood=0

    cdef int[:] x

    cdef dtype_t[::view.contiguous] buffer_1d = np.zeros(n_stats)
    cdef dtype_t[::view.contiguous] start_num = np.zeros(n_stats)
    cdef dtype_t[:,::view.contiguous] trans_num = np.zeros((n_stats, n_stats))
    cdef dtype_t[:,::view.contiguous] emission_num = np.zeros((n_stats, n_obs))



    cdef int n_x = data_slice.shape[0]
    cdef int max_x = 0;
    cdef double tmp_double = 0
    for k in range(n_x):
        # 观测序列的长度
        n_samples = data_slice[k][1] - data_slice[k][0]
        if n_samples > max_x:
            max_x = n_samples

    # 每个观测序列的长度不同，这里需要每次都重新申请空间 todo 是否可优化？
    cdef dtype_t[:,::view.contiguous] fwdlattice = np.zeros((max_x, n_stats))
    cdef dtype_t[:,::view.contiguous] bwdlattice = np.zeros((max_x, n_stats))
    cdef dtype_t[:,::view.contiguous] xi_sum = np.zeros((n_stats, n_stats))

    cdef dtype_t[::view.contiguous] gamma_sum = np.zeros(n_stats)
    cdef dtype_t[:,::view.contiguous] gamma = np.zeros((max_x, n_stats))


    for iter in range(max_iter):

        reset_zero_1d(start_num,n_stats)
        reset_zero_2d(trans_num,n_stats,n_stats)
        reset_zero_2d(emission_num,n_stats,n_obs)
        reset_zero_1d(gamma_sum,n_stats)
        reset_zero_2d(xi_sum,n_stats,n_stats)
        reset_zero_1d(buffer_1d,n_stats)

        # print("======",iter,"==========")
        # print_1d("start",startprob)
        # print_2d("transmat",transmat)
        # print_2d("emission",emissionprob)
        # print("ll",log_likelihood)
        with np.errstate(divide="ignore") :  # 忽略对0求log的警告
            log_startprob = np.log(startprob)
            log_transmat = np.log(transmat)
            log_emissionprob = np.log(emissionprob)

        log_likelihood = -INFINITY
        cur_log_likelihood = 0

        for k in range(n_x):
            # 观测序列的长度
            n_samples = data_slice[k][1] -  data_slice[k][0]
            if n_samples <= 1:
                continue
            x = data[data_slice[k][0]: data_slice[k][1]]
            # print(n_samples,x[0])
            # 计算前向算法，注意是加了log后的结果
            _forward(n_samples, n_stats,
                          log_startprob,
                          log_transmat,
                          log_emissionprob,
                          x, fwdlattice)

            # for t in range(100):
            #     print(fwdlattice[t][0],fwdlattice[t][1])
            # 累加对数似然值
            for i in range(n_stats):
                buffer_1d[i] = fwdlattice[n_samples-1,i]
            # 当前序列的对数似然
            tmp_double= _logsumexp(buffer_1d)
            log_likelihood += tmp_double

            # 计算后向算法，注意是加了log后的结果
            _backward(n_samples, n_stats,
                           log_startprob,
                           log_transmat,
                           log_emissionprob,x, bwdlattice)
            # 计算 xi 的累计和，没有log
            _compute_xi_sum(n_samples, n_stats, fwdlattice,
                                     log_transmat,
                                     bwdlattice, log_emissionprob,x,
                                     xi_sum,tmp_double)

            # 计算 gamma，
            for t in range(n_samples-1):
                # print("bwdlattice",bwdlattice[t,0],bwdlattice[t,1])
                tmp_double = 0
                for i in range(n_stats):
                    # print(t,i,fwdlattice[t][i],expl(fwdlattice[t][i]), bwdlattice[t][i],expl(bwdlattice[t][i]))
                    gamma[t][i] = expl(fwdlattice[t][i]) * expl(bwdlattice[t][i]) # 由于前后向已经加了log
                    tmp_double  += gamma[t][i]
                for i in range(n_stats):
                    gamma[t][i] /= tmp_double
                    gamma_sum[i] +=  gamma[t][i]



                # print("gamma",gamma[t,0],gamma[t,1])

            for i in range(n_stats):
                start_num[i] += gamma[0][i]  # 初始概率
                for j in range(n_stats):

                    trans_num[i][j] += xi_sum[i][j]  # 转移概率

            # 计算并累加发射概率
            for i in range(n_stats):
                for t in range(n_samples-1):
                    emission_num[i, x[t]] += gamma[t,i]

            # break

        # 综合所有序列的结果，求出最终的三元素
        for i in range(n_stats):
            startprob[i] = start_num[i] / n_x
            for j in range(n_stats):
                transmat[i][j] = trans_num[i][j] / gamma_sum[i]
                emissionprob[i][j] = emission_num[i][j] / gamma_sum[i]

        # print_1d("start_raw",startprob)
        # print_2d("transmat_raw",transmat)
        # print_2d("emission_raw",emissionprob)
        # print("ll",log_likelihood,cur_log_likelihood)
        # 约束条件
        bounded1d(startprob, startprob_lb, startprob_ub)
        bounded2d(transmat, transmat_lb, transmat_ub)
        bounded2d(emissionprob,emissionprob_lb, emissionprob_ub)

        if abs(cur_log_likelihood - log_likelihood) < tol:
            log_likelihood = cur_log_likelihood
            break
        log_likelihood = cur_log_likelihood

    return log_likelihood

cdef void bounded1d(dtype_t[:] source ,dtype_t[:] low,dtype_t[:] upper):
    cdef int i
    for i in range(source.shape[0]):
        if source[i] < low[i]:
            source[i] = low[i]
        if source[i] > upper[i]:
            source[i] = upper[i]

cdef void bounded2d(dtype_t[:,:] source ,dtype_t[:,:] low,dtype_t[:,:] upper):
    cdef int i,j
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            if source[i,j] < low[i,j]:
                source[i,j] = low[i,j]
            if source[i,j] > upper[i,j]:
                source[i,j] = upper[i,j]

cdef void print_1d(name,dtype_t[:] array):
    print(name)
    for i in range(array.shape[0]):
        print(array[i])



cdef void print_2d(name,dtype_t[:,:] array):
    print(name)
    for i in range(array.shape[0]):
        # for j in range(array.shape[1]):
        print(array[i,0],array[i,1])

        # print("")