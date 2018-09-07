# from cython cimport view
cimport cython
import numpy as np
cimport numpy as np
from numpy.math cimport isinf, fabsl, INFINITY, isnan as np_isnan
# from numpy.random cimport norm
# from scipy.stats import norm
ctypedef np.float_t FLOAT_T
# import random
ctypedef np.int64_t INT_T
from libc.math cimport exp, sqrt, log,pi
# from libc.stdlib cimport rand, RAND_MAX
# from cpython cimport array
# import array
#from cython_gsl cimport *
# from talirt.model.crandom cimport uniform_rv,normal_rv
# DEF CONSTANT_D=1.0



cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)

cdef extern from "gsl/gsl_randist.h":
    double _gamma "gsl_ran_gamma"(gsl_rng * r,double,double) nogil
    double _gaussian "gsl_ran_gaussian"(gsl_rng * r,double) nogil
    double _gaussian_pdf "gsl_ran_gaussian_pdf"(double,double) nogil
    double _uniform "gsl_rng_uniform"(gsl_rng * r) nogil

cdef gsl_rng *_gsl_r = gsl_rng_alloc(gsl_rng_mt19937)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int _u1irt(
        np.float_t[:,:] ret,
        np.float_t[:] theta,
        np.float_t[:] slope,
        np.float_t[:] intercept,
        np.float_t[:] guess,
        int user_count,
        int item_count
           ) nogil :
    cdef int i=0
    cdef int j = 0
    cdef double z = 0

    for i in range(user_count):
        for j in range(item_count):
            # z = theta[i] - intercept[j]
            # 论文的版本
            z = theta[i] + intercept[j]
            ret[i,j] = 1.0 / (1.0 + exp(-z))
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int _u2irt(
        np.float_t[:,:] ret,
        np.float_t[:] theta,
        np.float_t[:] slope,
        np.float_t[:] intercept,
        np.float_t[:] guess,
        int user_count,
        int item_count
           ) nogil :

    cdef int i=0
    cdef int j = 0
    cdef double z=0
    for i in range(user_count):
        for j in range(item_count):
            # z = slope[j] * (theta[i] - intercept[j])
            # 论文的版本
            z = slope[j] * theta[i] + intercept[j]
            ret[i,j] =  1.0 / (1.0 + exp(-z))
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int _u3irt(
        np.float_t[:,:] ret,
        np.float_t[:] theta,
        np.float_t[:] slope,
        np.float_t[:] intercept,
        np.float_t[:] guess,
        int user_count,
        int item_count
           ) nogil :

    cdef int i=0
    cdef int j = 0
    # cdef int n = theta.size
    # cdef int m = slope.size
    cdef double z
    # cdef np.ndarray data=np.zeros(shape=(n,m))
    # cdef np.float_t[:,:] data_ptr = data

    for i in range(user_count):
        for j in range(item_count):
            # z = slope[j] * (theta[i] - intercept[j])
            # 论文的版本
            z = slope[j] * theta[i] + intercept[j]
            ret[i,j] = guess[j] + (1 - guess[j]) * (1.0 / (1.0 + exp(-z)))
    return 0





@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef inline double _log_likelihood(np.float_t[:] theta,
                       np.float_t[:] slope,
                       np.float_t[:] intercept,
                       np.float_t[:] guess,
                       np.float_t[:,:] response,
                            int user_count,
                            int item_count
                             ) nogil:

    cdef int i=0
    cdef int j = 0
    cdef double irt=0,z=0
    cdef double result=0,tmp=0
    # cdef int n = theta.size
    # cdef int m = intercept.size

    for i in range(user_count):
        for j in range(item_count):
            # 作答记录里存在空值
            if np_isnan(response[i][j]):
                continue
            # z = slope[j] * (theta[i] - intercept[j])
            # 论文的版本
            z = slope[j] * theta[i] + intercept[j]
            irt = guess[j] + (1.0 - guess[j]) * (1.0 / (1 + exp(-z)))
            tmp = response[i][j]*log(irt) + (1-response[i][j])*log(1-irt)
            result +=tmp
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int _log_likelihood_user(
        np.float_t[:] ret,
        np.float_t[:,:] response,
        np.float_t[:] theta,
        np.float_t[:] slope,
        np.float_t[:] intercept,
        np.float_t[:] guess,
        int user_count,
        int item_count
                    ) nogil:

    cdef int i=0
    cdef int j = 0
    cdef double irt=0,z=0
    cdef double result=0,tmp=0
    # 按照人循环
    for i in range(user_count):
        # 按照题目循环
        for j in range(item_count):
            # 作答记录里存在空值
            if np_isnan(response[i][j]):
                continue
            # z = slope[j] * (theta[i] - intercept[j])
            # 论文的版本
            z = slope[j] * theta[i] + intercept[j]
            irt = guess[j] + (1 - guess[j]) * (1 / (1 + exp(-z)))
            tmp = response[i][j]*log(irt) + (1-response[i][j])*log(1-irt)
            # result +=tmp
            ret[i] += tmp
    return 0




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.nonecheck(False)
cdef int _sample_theta(
                np.float_t[:] ret,
                double theta,
                np.float_t[:] slope,
                np.float_t[:] intercept,
                np.float_t[:] guess,
                np.float_t[:,:] response,
                int item_count,
                np.float_t[:] cache1,
                np.float_t[:] cache2,
                unsigned int burn_in=10,
                unsigned int n=60,
                np.float_t mean=0,
                np.float_t sigma=1,
                ) nogil:



    cdef unsigned int iter = 0  # 总的抽样次数
    cdef unsigned int index = 0

    cdef double r=0
    cdef double v1=0
    cdef double v2=0

    # pre_theta[0] = theta
    cache1[0] = theta
    v1 = _log_likelihood(theta=cache1,
                         slope=slope, intercept=intercept, guess=guess,
                         response=response,user_count=1,item_count=item_count) + log(_gaussian_pdf(cache1[0],1))


    while iter < n:
        cache2[0] = _gaussian(_gsl_r,sigma)+cache1[0]
        # next_ptr[0] = next_theta
        v2 = _log_likelihood(theta=cache2,
                             slope=slope, intercept=intercept, guess=guess,
                             response=response,user_count=1,item_count=item_count) + log(_gaussian_pdf(cache2[0],1))
        r=exp(v2 - v1)
        # print(iter,pre_theta,next_theta,r)
        if r >= 1 or (r>0.1 and  _uniform(_gsl_r) <= r):
            iter += 1
            if iter > burn_in:
                ret[index] = cache2[0]
                index += 1
            cache1[0] = cache2[0]
            # pre_ptr[0]=pre_theta
            v1 = v2

    # return data
    return 0

############################
# 导出到python层的函数
############################

def gaussian_rvs(sd=1):
     return _gaussian(_gsl_r,sd)

def uniform_rvs():
     return _uniform(_gsl_r)
def gaussian_pdf(x,sd=1):
    return _gaussian_pdf(x,sd)

def sample_theta(theta,slope,intercept,guess,response,burn_in=10,n=60):


    cdef np.ndarray data =np.zeros(n - burn_in)
    cdef item_count= intercept.size
    cdef cache1 = np.zeros(1)
    cdef cache2 = np.zeros(1)
    _sample_theta(data,theta,
                    slope.flatten(),
                    intercept.flatten(),
                    guess.flatten(),
                    response,
                    item_count,
                  cache1,
                  cache2,
                    burn_in,n)
    return data

def log_likelihood(np.float_t[:,:] response,np.float_t[:] theta,
                       np.float_t[:] slope,
                       np.float_t[:] intercept,
                       np.float_t[:] guess,
                    ):
    """
    .. math::
            Object function  = - \ln L(x;\theta)=-(\sum_{i=0}^n ({y^{(i)}} \ln P + (1-y^{(i)}) \ln (1-P)))
    Parameters
    ----------
    theta
    a
    b
    c

    Returns
    -------
        res : float

    """
    cdef int user_count=theta.size
    cdef int item_count =intercept.size
    cdef np.float_t result = _log_likelihood(
        theta=theta,
        slope=slope,
        intercept=intercept,
        guess=guess,
        response=response,
        user_count=user_count,
        item_count=item_count)
    # print(result)
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def  log_likelihood_user(np.float_t[:,:] response,np.float_t[:] theta,
                       np.float_t[:] slope,
                       np.float_t[:] intercept,
                       np.float_t[:] guess
                             ):
    cdef int user_count=theta.size
    cdef int item_count = intercept.size
    ret = np.zeros(user_count)
    _log_likelihood_user(ret=ret,
                         response=response,
                         theta=theta,
                         slope=slope,intercept=intercept,guess=guess,
                         user_count=user_count,
                         item_count=item_count
                         )

    return ret


def u2irt_item_jac_and_hessian(np.float_t[:] theta,
                       np.float_t[:] slope,
                       np.float_t[:] intercept,
                       np.float_t[:] guess,
                     np.float_t[:,:] response):
    """
    三参数模型的一阶导数和二阶导数。z=slope*(theta-intercept)
    Parameters
    ----------
    response
    a
    b
    c

    Returns
    -------

    """
    cdef int i=0
    cdef int j = 0
    cdef n = theta.size
    cdef m = intercept.size
    cdef np.float_t error = 0
    cdef y_hat = np.zeros((n,m))
    _u3irt(ret=y_hat,theta=theta,slope=slope,intercept=intercept,guess=guess,user_count=n,item_count=m)

    cdef np.ndarray grad_1 = np.zeros((m,2))
    cdef float[:,:] grad_1_ptr = grad_1


    cdef np.ndarray grad_2 = np.zeros((m,2,2))
    cdef float[:,:,:] grad_2_ptr = grad_2

    for i in range(n):
        for j in range(m):
            # 作答记录里存在空值
            if np_isnan(response[i][j]):
                continue
            error = response[i][j] - y_hat[i][j]
            grad_1_ptr[j,0] +=  error*theta[i]  # 题目j的参数a 区分度的一阶导数
            grad_1_ptr[j,1] +=  error  # 题目j的参数b 难度的一阶导数
            # grad_1_ptr[j,2] +=  error*slope[j]  # 题目j的参数guess的一阶导数，c的导数很复杂，需要二参数irt的预测值

            # 二阶导数有个负号
            grad_2_ptr[j,0,0] += y_hat[i][j]*(1-y_hat[i][j])*theta[i]*theta[i]
            grad_2_ptr[j,1,1] += y_hat[i][j]*(1-y_hat[i][j])

            grad_2_ptr[j,0,1] += y_hat[i][j]*(1-y_hat[i][j])*theta[i]
            grad_2_ptr[j,1,0] = grad_2_ptr[j,0,1]

    return grad_1, -grad_2


def u2irt_item_jac(np.float_t[:] theta,
                       np.float_t[:] slope,
                       np.float_t[:] intercept,
                       np.float_t[:] guess,
                     np.float_t[:,:] response):
    """

    Parameters
    ----------
    response
    a
    b
    c

    Returns
    -------

    """
    cdef int i=0
    cdef int j = 0
    cdef n = theta.size
    cdef m = intercept.size
    cdef np.float_t error = 0
    cdef y_hat = np.zeros((n,m))
    _u3irt(ret=y_hat,theta=theta,slope=slope,intercept=intercept,guess=guess,user_count=n,item_count=m)

    cdef np.ndarray grad_1 = np.zeros((m,2))
    cdef float[:,:] grad_1_ptr = grad_1

    for i in range(n):
        for j in range(m):
            # 作答记录里存在空值
            if np_isnan(response[i][j]):
                continue
            error = response[i][j] - y_hat[i][j]
            grad_1_ptr[j,0] += error*theta[i]  # 题目j的参数a 区分度的一阶导数
            grad_1_ptr[j,1] += error  # 题目j的参数b 难度的一阶导数


    return grad_1

def uirt_theta_jac(np.float_t[:] theta,
                       np.float_t[:] slope,
                       np.float_t[:] intercept,
                       np.float_t[:] guess,
                     np.float_t[:,:] response):
    """

    Parameters
    ----------
    theta
    a
    b
    c
    response

    Returns
    -------

    """
    cdef int i=0
    cdef int j = 0
    cdef n = theta.shape[0]
    cdef m = slope.shape[0]
    cdef double error = 0
    cdef y_hat = np.zeros((n,m))
    _u3irt(ret=y_hat,theta=theta,slope=slope,intercept=intercept,guess=guess,user_count=n,item_count=m)
    cdef np.ndarray grad = np.zeros(n)
    cdef double [:] grad_ptr = grad
    # cdef np.float_t grade=0
    for i in range(n):
        for j in range(m):
            # 作答记录里存在空值
            if np_isnan(response[i][j]):
                continue
            error = response[i][j] - y_hat[i][j]
            grad_ptr[i] += error*slope[j]
    return grad


@cython.boundscheck(False)
@cython.wraparound(False)
def  uirt_matrix(
        theta,
        slope,
        intercept,
        guess=None
           ):

    cdef int user_count=theta.size
    cdef int item_count = intercept.size
    ret = np.zeros((user_count,item_count))
    if slope is None:
        _u1irt(ret=ret,theta=theta,slope=slope,intercept=intercept,guess=guess,user_count=user_count,item_count=item_count)
    elif guess is None:

        _u2irt(ret=ret,theta=theta,slope=slope,intercept=intercept,guess=guess,user_count=user_count,item_count=item_count)
    else:
        _u3irt(ret=ret,theta=theta,slope=slope,intercept=intercept,guess=guess,user_count=user_count,item_count=item_count)

    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
def u3irt_sequence(
        np.float_t[:] theta,
        np.float_t[:] slope,
        np.float_t[:] intercept,
        np.float_t[:] guess
    ) :

    cdef int i=0
    cdef int n = theta.size

    cdef np.ndarray data=np.zeros(shape=(n,))
    cdef np.float_t[:] data_ptr = data

    for i in range(n):
            z = slope[i] * theta[i] + intercept[i]
            # z = a[i] * (theta[i] + b[i])
            data_ptr[i] = guess[i] + (1 - guess[i]) * (1 / (1 + exp(-z)))
    return data


@cython.boundscheck(False)
@cython.wraparound(False)
def u2irt_sequence(
        np.float_t[:] theta,
        np.float_t[:] slope,
        np.float_t[:] intercept
           ) :

    cdef int i=0
    cdef int n = theta.size

    cdef np.ndarray data=np.zeros(shape=(n,))
    cdef np.float_t[:] data_ptr = data

    for i in range(n):
            # z = D * slope[j] * (theta[i] - intercept[j])
            z = slope[i] *theta[i] + intercept[i]
            data_ptr[i] =   1.0 / (1.0 + exp(-z))
    return data

@cython.boundscheck(False)
@cython.wraparound(False)
def u1irt_sequence(
        np.float_t[:] theta,
        np.float_t[:] intercept
           ) :

    cdef int i=0
    cdef int n = theta.size
    cdef np.ndarray data=np.zeros(shape=(n,))
    cdef np.float_t[:] data_ptr = data

    for i in range(n):
            # z = D * slope[j] * (theta[i] - intercept[j])
            z =  theta[i] + intercept[i]
            data_ptr[i] =   1.0 / (1.0 + exp(-z))
    return data
