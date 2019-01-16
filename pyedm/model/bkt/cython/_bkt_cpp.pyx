# distutils: language = c++
cimport cython
import numpy as np
cimport numpy as np

cdef extern from "_bkt.cpp":
    pass

cdef extern from "_bkt.h":
    cdef cppclass StandardBKT:
        StandardBKT(int, int) except +
        void init(double *pi, double *a, double *b);
        void set_bound_pi(double *lower, double *upper);
        void set_bound_a(double *lower, double *upper);
        void set_bound_b(double *lower, double *upper);
        double estimate(int *x, int *lengths, int n_lengths, int max_iter, double tol);
        double get_pi(double *out);
        double get_a(double *out);
        double get_b(double *out);
        void predict_next(double *out, int *x, int n_x, double *pi, double *a, double *b, int n_stat, int n_obs);
        int iter;
        double log_likelihood;
        int n_obs;
        int n_stat;
        int max_iter;


ctypedef double dtype_t

cdef void *get_pointer(np.ndarray arr):
    if arr is None:
        return NULL
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    # cdef double[::1] lower_view = arr
    return arr.data

cdef class SBKT:
    cdef StandardBKT *c_object
    cdef int n_stat
    cdef int n_obs
    def __cinit__(self, int n_stat=2, int n_obs=2):
        self.c_object = new StandardBKT(n_stat, n_obs)
        self.n_stat = n_stat
        self.n_obs = n_obs

    def init(self, np.ndarray[double,ndim=1] start=None, np.ndarray[double,ndim=2] transition=None, np.ndarray[double,ndim=2] emission=None):
        """
        初始化设置参数
        Parameters
        ----------
        start  初始矩阵
        transition  转移矩阵
        emission 发射矩阵

        Returns
        -------

        """
        self.c_object.init(<double*> get_pointer(start), <double*> get_pointer(transition),
                           <double*> get_pointer(emission))

    def set_bounded_start(self, np.ndarray[double,ndim=1] lower=None, np.ndarray[double,ndim=1] upper=None):
        self.c_object.set_bound_pi(<double *> get_pointer(lower), <double *> get_pointer(upper))
        # self.c_object.setBoundedPI(<double *> &lower[0], <double *> &upper[0])

    def set_bounded_transition(self, np.ndarray[double,ndim=2] lower=None, np.ndarray[double,ndim=2] upper=None):
        self.c_object.set_bound_a(<double *> get_pointer(lower), <double *> get_pointer(upper))
        # self.c_object.setBoundedA(<double *> &lower[0][0], <double *> &upper[0][0])

    def set_bounded_emission(self, np.ndarray[double,ndim=2] lower=None, np.ndarray[double,ndim=2] upper=None):
        self.c_object.set_bound_b(<double *> get_pointer(lower), <double *> get_pointer(upper))
        # self.c_object.setBoundedB(<double *> &lower[0][0], <double *> &lower[0][0])

    def estimate(self, np.ndarray[int, ndim=1] x, np.ndarray[int, ndim=1] lengths, int max_iter = 20,
                 double tol = 1e-2):
        """

        Parameters
        ----------
        x
        lengths
        max_iter
        tol

        Returns
        -------

        """
        return self.c_object.estimate(<int*> get_pointer(x), <int*> get_pointer(lengths), lengths.shape[0], max_iter,
                                      tol)
    def predict_next(self, int[::1] x, int n_x,
                     np.ndarray[double,ndim=1] start=None,
                     np.ndarray[double,ndim=2] transition=None,
                     np.ndarray[double,ndim=2] emission=None,
                     int n_stat=0, int n_obs=0):
        """
        预测下一个观测值
        Parameters
        ----------
        x  已知的观测序列
        n_x  已知观测序列的长度
        start  初始概率值
        transition  转移概率矩阵
        emission  发射概率矩阵
        n_stat  隐状态的数量
        n_obs  观测状态的数量

        Returns
        -------

        """

        out = np.zeros(n_obs if n_obs > 0 else self.n_obs, dtype=np.float64)
        self.c_object.predict_next(<double*> get_pointer(out), <int*> &x[0], n_x,
                                   <double*> get_pointer(start),
                                   <double*> get_pointer(transition),
                                   <double*> get_pointer(emission),
                                   n_stat,n_obs)
        return out
        pass
    @property
    def start(self):
        """
        初始矩阵
        Returns
        -------

        """
        pi = np.zeros(self.n_stat, dtype=np.float64)
        self.c_object.get_pi(<double*> get_pointer(pi))
        return pi

    @property
    def transition(self):
        """
        转移矩阵
        Returns
        -------

        """
        arr = np.zeros(shape=(self.n_stat, self.n_stat), dtype=np.float64)
        self.c_object.get_a(<double*> get_pointer(arr))
        return arr

    @property
    def emission(self):
        """
        发射矩阵
        Returns
        -------

        """
        arr = np.zeros(shape=(self.n_stat, self.n_obs), dtype=np.float64)
        self.c_object.get_b(<double*> get_pointer(arr))
        return arr

    @property
    def iter(self):
        return self.c_object.iter

    @property
    def log_likelihood(self):
        return self.c_object.log_likelihood

    def __dealloc__(self):
        del self.c_object

cdef class IRTBKT(SBKT):
    pass
