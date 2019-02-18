# distutils: language = c++
# _distutils: sources = _bkt/_bkt.cpp
# _distutils: include_dirs = _bkt/
cimport cython
import numpy as np
cimport numpy as np

# from cython.parallel import prange,parallel
# cimport openmp
# from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libcpp cimport bool


# from libcpp cimport new,delete
cdef extern from "_bkt/hmm.cpp":
    pass
cdef extern from "_bkt/hmm.h" nogil:
    cdef cppclass _HMM "HMM":
        _HMM(int, int) nogil except +
        void init(double *pi, double *a, double *b) nogil;
        void set_bound_pi(double *lower, double *upper) nogil;
        void set_bound_a(double *lower, double *upper) nogil;
        void set_bound_b(double *lower, double *upper) nogil;
        int fit(int *x, int *lengths, int n_lengths, int max_iter, double tol) nogil;
        void get_pi(double *out) nogil;
        void get_a(double *out) nogil;
        void get_b(double *out) nogil;
        double predict_by_posterior(double *out, int *x, int n_x) nogil;
        double posterior_distributed(double *out, int *x, int n_x) nogil;
        double viterbi(int *out, int *x, int n_x);
        void predict_by_viterbi(double *out, int *x, int n_x);
        void predict_first(double *out);
        void set_minimum_obs(int value);
        int get_minimum_obs();
        int iter;
        double log_likelihood;
        int n_obs;
        int n_stat;
        int max_iter;
        bool success;

cdef extern from "_bkt/bkt.cpp":
    pass


cdef extern from "_bkt/bkt.h" nogil:
    cdef cppclass _StandardBKT "StandardBKT":
        _StandardBKT(int, int) nogil except +
        void init(double *pi, double *a, double *b) nogil;
        void set_bound_pi(double *lower, double *upper) nogil;
        void set_bound_a(double *lower, double *upper) nogil;
        void set_bound_b(double *lower, double *upper) nogil;
        int fit(int *x, int *lengths, int n_lengths, int max_iter, double tol) nogil;
        void get_pi(double *out) nogil;
        void get_a(double *out) nogil;
        void get_b(double *out) nogil;
        # void predict_next(double *out, int *x, int n_x) nogil;
        double posterior_distributed(double *out, int *x, int n_x) nogil;
        double viterbi(int *out, int *x, int n_x);
        double predict_by_posterior(double *out, int *x, int n_x);
        double predict_by_viterbi(double *out, int *x, int n_x);
        void predict_first(double *out);
        void set_minimum_obs(int value);
        int get_minimum_obs();
        int iter;
        double log_likelihood;
        int n_obs;
        int n_stat;
        int max_iter;
        bool success;

    cdef cppclass _IRTBKT "IRTBKT":
        _IRTBKT(int, int) nogil except +
        # void set_items_data(double *items, int length) nogil;
        void set_items_info(double *items, int length) nogil;
        void set_obs_items(int *items_id, int length) nogil;
        void init(double *pi, double *a, double *b) nogil;
        void set_bound_pi(double *lower, double *upper) nogil;
        void set_bound_a(double *lower, double *upper) nogil;
        void set_bound_b(double *lower, double *upper) nogil;
        int fit(int *x, int *lengths, int n_lengths, int max_iter, double tol) nogil;
        void get_pi(double *out) nogil;
        void get_a(double *out) nogil;
        void get_b(double *out) nogil;
        double predict_by_posterior(double *out, int *x, int n_x, int item_id) nogil;
        double posterior_distributed(double *out, int *x, int n_x) nogil;
        double viterbi(int *out, int *x, int n_x);
        double predict_by_viterbi(double *out, int *x, int n_x, int item_id);
        void predict_first(double *out, int item_id);
        void set_minimum_obs(int value);
        int get_minimum_obs();
        void debug();
        int iter;
        double log_likelihood;
        int n_obs;
        int n_stat;
        int max_iter;
        bool success;



cdef extern from "_bkt/TrainHelper.cpp":
    pass
cdef extern from "_bkt/TrainHelper.h" nogil:
    cdef cppclass _TrainHelper "TrainHelper":
        _TrainHelper(int n_stat, int n_obs, int model_type) except +
        void init(double *pi, double *a, double *b, bool copy)
        void set_bound_pi(double *lower, double *upper, bool copy)
        void set_bound_a(double *lower, double *upper, bool copy)
        void set_bound_b(double lower[], double upper[], bool copy)
        void set_items_info(double items[], int length, bool copy)
        void fit(int trace[], int group[], int x[], int length, int item[], int max_iter, double tol)
        _HMM ** models
        int model_count

ctypedef double dtype_t

cdef void *get_pointer(np.ndarray arr):
    if arr is None:
        return NULL

    if not arr.flags['C_CONTIGUOUS']:
        # 注意如果触发了这步，重新生成了一个局部变量，返回的指针会被python自动回收，不安全。！！！
        # raise ValueError("np.ndarray must be contiguousarray. you can arr = np.ascontiguousarray(arr) ")
        arr = np.ascontiguousarray(arr)
    # cdef double[::1] lower_view = arr
    return arr.data

# cdef class _Model:


cdef class StandardBKT:
    """
    标准BKT模型
    """
    cdef _HMM *c_object
    cdef int n_stat
    cdef int n_obs

    def __dealloc__(self):
        del self.c_object

    def __cinit__(self, int no_object=0, *argv, **kwargs):

        if self.c_object != NULL:
            del self.c_object

        if no_object == 0:
            self.c_object = <_HMM*> new _StandardBKT(2, 2)
        else:
            self.c_object = NULL

    def __init__(self, int no_object=0):
        self.n_stat = 2
        self.n_obs = 2

        if self.c_object == NULL:
            return

        start_init = np.array([1.0 / self.n_stat] * self.n_stat, dtype=np.float64)
        # assert start_init.sum() == 1
        start_lb = np.array([0] * self.n_stat, dtype=np.float64)
        start_ub = np.array([1] * self.n_stat, dtype=np.float64)

        transition_init = np.array([[0.4, 0.6], [0, 1]], dtype=np.float64)
        transition_lb = np.array([[0, 0], [0, 1]], dtype=np.float64)
        transition_ub = np.array([[1, 1], [0, 1]], dtype=np.float64)
        emission_init = np.array([[0.8, 0.2], [0.2, 0.8]], dtype=np.float64)

        emission_lb = np.array([[0.7, 0], [0, 0.7]], dtype=np.float64)
        emission_ub = np.array([[1, 0.3], [0.3, 1]], dtype=np.float64)
        self.init(start_init, transition_init, emission_init)
        self.set_bounded_start(start_lb, start_ub)
        self.set_bounded_transition(transition_lb, transition_ub)
        self.set_bounded_emission(emission_lb, emission_ub)

    def init(self, np.ndarray[cython.floating, ndim=1] start=None,
             np.ndarray[cython.floating, ndim=2] transition=None,
             np.ndarray[cython.floating, ndim=2] emission=None):
        """
        初始化设置参数
        Parameters
        ----------
        start : np.ndarray[double,ndim=1] shape=(2,)
            初始矩阵
        transition : np.ndarray[double,ndim=2] shape=(2,2)
            转移矩阵
        emission : np.ndarray[double,ndim=2] shape=(2,2)
            发射矩阵

        Returns
        -------

        """
        if start is not None:
            assert abs(start.sum() - 1.0) < 1e-12
            start = np.ascontiguousarray(start, dtype=np.float64)

        if transition is not None:
            assert np.all(abs(1.0 - transition.sum(1)) < 1e-12)
            transition = np.ascontiguousarray(transition, dtype=np.float64)

        if emission is not None:
            assert np.all(abs(1.0 - emission.sum(1)) < 1e-12)
            emission = np.ascontiguousarray(emission, dtype=np.float64)

        if self.c_object != NULL:
            self.c_object.init(<double*> get_pointer(start), <double*> get_pointer(transition),
                               <double*> get_pointer(emission))

    def set_bounded_start(self, np.ndarray[cython.floating, ndim=1] lower=None,
                          np.ndarray[cython.floating, ndim=1] upper=None):
        """
        设置初始概率矩阵的约束
        Parameters
        ----------
        lower : np.ndarray[double,ndim=1] shape=(2,)
            下限约束
        upper : np.ndarray[double,ndim=1] shape=(2,)
            上限约束
        Returns
        -------

        """
        if lower is not None:
            lower = np.ascontiguousarray(lower, dtype=np.float64)
        if upper is not None:
            upper = np.ascontiguousarray(upper, dtype=np.float64)

        if self.c_object != NULL:
            self.c_object.set_bound_pi(<double *> get_pointer(lower), <double *> get_pointer(upper))
        # self.c_object.setBoundedPI(<double *> &lower[0], <double *> &upper[0])

    def set_bounded_transition(self, np.ndarray[cython.floating, ndim=2] lower=None,
                               np.ndarray[cython.floating, ndim=2] upper=None):
        """
        设置转移概率矩阵的约束
        Parameters
        ----------
        lower : np.ndarray[double,ndim=2] shape=(2,2)
            下限约束
        upper : np.ndarray[double,ndim=2] shape=(2,2)
            上限约束

        Returns
        -------

        """
        if lower is not None:
            lower = np.ascontiguousarray(lower, dtype=np.float64)
        if upper is not None:
            upper = np.ascontiguousarray(upper, dtype=np.float64)

        if self.c_object != NULL:
            self.c_object.set_bound_a(<double *> get_pointer(lower), <double *> get_pointer(upper))
        # self.c_object.setBoundedA(<double *> &lower[0][0], <double *> &upper[0][0])

    def set_bounded_emission(self, np.ndarray[double, ndim=2] lower=None, np.ndarray[double, ndim=2] upper=None):
        """
        设置发射概率矩阵的约束
        Parameters
        ----------
        lower : np.ndarray[double,ndim=2] shape=(2,2)
            下限约束
        upper : np.ndarray[double,ndim=2] shape=(2,2)
            上限约束

        Returns
        -------

        """
        if lower is not None:
            lower = np.ascontiguousarray(lower, dtype=np.float64)
        if upper is not None:
            upper = np.ascontiguousarray(upper, dtype=np.float64)

        if self.c_object != NULL:
            self.c_object.set_bound_b(<double *> get_pointer(lower), <double *> get_pointer(upper))
        # self.c_object.setBoundedB(<double *> &lower[0][0], <double *> &lower[0][0])

    def fit(self, np.ndarray[cython.integral, ndim=1] x, np.ndarray[cython.integral, ndim=1] lengths, int max_iter = 20,
            double tol = 1e-2):
        """
        训练模型
        Parameters
        ----------
        x : np.ndarray[int, ndim=1]
            观测数据，如果是多个独立观测序列，首尾衔接，穿在一起。每个观测序列的长度通过参数lengths指定。
        lengths : np.ndarray[int, ndim=1]
            每个观测序列的长度
        max_iter : int
            最大迭代次数
        tol

        Returns
        -------

        """
        if self.c_object == NULL:
            return None

        assert x is not None
        assert lengths is not None

        x = np.ascontiguousarray(x, dtype=np.int32)
        lengths = np.ascontiguousarray(lengths, dtype=np.int32)

        cdef int *x_ptr = <int*> get_pointer(x)
        cdef int *l_ptr = <int*> get_pointer(lengths)
        cdef int ll = lengths.shape[0]
        cdef int ret = 0
        with nogil:
            ret = self.c_object.fit(x_ptr, l_ptr, ll, max_iter, tol)

        return ret

    def predict_next(self, np.ndarray[cython.integral, ndim=1] x=None, str algorithm="viterbi"):
        """
        预测下一个观测值，两种算法 viterbi 和 posterior(后验概率分布)。
        Parameters
        ----------
        x : np.ndarray[int, ndim=1]
            已知的观测序列
        algorithm : str
            指定采用的算法，包括算法 viterbi 和 posterior(后验概率分布)。['viterbi','posterior']

        Returns
        -------
            np.ndarray[int, ndim=1] shape=(2,) 观测状态的概率分布
        """
        out = np.zeros(self.n_obs, dtype=np.float64, order="C")
        cdef int n_x = 0
        # 预测首次作答结果
        if x is None or x.shape[0] == 0:
            x = None
            n_x = 0
        else:
            n_x = x.shape[0]
            x = np.ascontiguousarray(x, dtype=np.int32)

        if algorithm == 'viterbi':
            if n_x > 0:
                (<_StandardBKT*> self.c_object).predict_by_viterbi(<double*> get_pointer(out), <int*> get_pointer(x),
                                                                   n_x)
        elif algorithm == "posterior":
            (<_StandardBKT*> self.c_object).predict_by_posterior(<double*> get_pointer(out), <int*> get_pointer(x), n_x)
        else:
            raise ValueError("Unknown algorithm:%s" % algorithm)
        return out

    def posterior_distributed(self, np.ndarray[cython.integral, ndim=1] x):
        """
        计算后验概率分布
        Parameters
        ----------
        x : np.ndarray[int, ndim=1]
            已知的观测序列

        Returns
        -------
            np.ndarray[double, ndim=2] shape=(n_x,n_stat)
            返回每个观测值的对应隐状态的后验概率分布

        """

        cdef double ll;
        cdef int n_x = x.shape[0]

        assert x is not None

        x = np.ascontiguousarray(x, dtype=np.int32)

        out = np.zeros((n_x, self.n_stat), dtype=np.float64, order='C')

        (<_StandardBKT*> self.c_object).posterior_distributed(<double*> get_pointer(out), <int*> get_pointer(x), n_x)
        return out

    def viterbi(self, np.ndarray[cython.integral, ndim=1] x):
        """
        viterbi 解码算法
        Parameters
        ----------
        x : np.ndarray[int, ndim=1]
            已知的观测序列
        Returns
        -------
            np.ndarray[int, ndim=1] shape=(n_x)
            返回最有隐状态序列。
        """
        assert x is not None

        cdef double prob;
        cdef int n_x = x.shape[0]

        x = np.ascontiguousarray(x, dtype=np.int32)

        out = np.zeros(n_x, dtype=np.int32, order='C')

        prob = (<_StandardBKT*> self.c_object).viterbi(<int*> get_pointer(out), <int*> get_pointer(x), n_x)

        return out

    @property
    def start(self):
        """
        返回初始概率矩阵
        Returns
        -------
            np.ndarray[double, ndim=1] shape=(n_stat)
        """
        pi = np.zeros(self.n_stat, dtype=np.float64, order='C')
        self.c_object.get_pi(<double*> get_pointer(pi))
        return pi

    @property
    def transition(self):
        """
        返回转移矩阵
        Returns
        -------
            np.ndarray[double, ndim=2] shape=(n_stat,n_stat)
        """
        arr = np.zeros(shape=(self.n_stat, self.n_stat), dtype=np.float64, order='C')
        self.c_object.get_a(<double*> get_pointer(arr))
        return arr

    @property
    def emission(self):
        """
        返回发射矩阵
        Returns
        -------
            np.ndarray[double, ndim=2] shape=(n_stat,2)
        """
        arr = np.zeros(shape=(self.n_stat, self.n_obs), dtype=np.float64, order='C')
        self.c_object.get_b(<double*> get_pointer(arr))
        return arr

    @property
    def iter(self):
        return self.c_object.iter

    @property
    def log_likelihood(self):
        return self.c_object.log_likelihood

    @property
    def success(self):
        """
        模型是否训练成功
        Returns
        -------

        """
        return self.c_object.success

    @property
    def minimum_obs(self):
        """
        最小数量的观测序列长度，小于这个长度的观测序列无法进行训练
        Returns
        -------

        """
        return self.c_object.get_minimum_obs()

    def set_minimum_obs(self, int value):
        self.c_object.set_minimum_obs(value)

    def show(self):

        print('success:', self.c_object.success, 'n_stat=%d' % self.n_stat, "n_obs=%d" % self.n_obs)
        print('n_stat=%d' % self.n_stat, "n_obs=%d" % self.n_obs)
        print("-" * 10, "start", "-" * 10)
        print(self.start.round(4))
        print("-" * 10, "transition", "-" * 10)
        print(self.transition.round(4))
        print("-" * 10, "emission", "-" * 10)
        print(self.emission.round(4))

cdef class IRTBKT(StandardBKT):
    """
    IRT变种BKT

    """
    cdef np.ndarray items_info
    cdef np.ndarray train_items

    def __cinit__(self, int n_stat=7, int no_object=0, *argv, **kwargs):
        #     # print(n_stat)
        if self.c_object != NULL:
            del self.c_object
        if no_object == 0:
            self.c_object = <_HMM*> new _IRTBKT(n_stat, 2)
        else:
            self.c_object = NULL

    def __init__(self, int n_stat=7, int no_object=0):
        # super(IRTBKT,self).__init__(n_stat,n_obs)
        self.items_info = None
        self.train_items = None
        self.n_stat = n_stat
        self.n_obs = 2

        if self.c_object == NULL:
            return
        start_init = np.array([1.0 / n_stat] * n_stat, dtype=np.float64)
        # assert start_init.sum() == 1
        start_lb = np.array([0] * n_stat, dtype=np.float64)
        start_ub = np.array([1] * n_stat, dtype=np.float64)
        start_ub = np.asarray([0.5, 1, 1, 1, 0.5, 0.3, 0.1])
        if n_stat == 7:
            transition_init = np.array([
                [0.5, 0.5, 0, 0, 0, 0, 0],
                [0, 0.5, 0.5, 0, 0, 0, 0],
                [0, 0, 0.5, 0.5, 0, 0, 0],
                [0, 0, 0, 0.5, 0.5, 0, 0],
                [0, 0, 0, 0, 0.5, 0.5, 0],
                [0, 0, 0, 0, 0, 0.5, 0.5],
                [0, 0, 0, 0, 0, 0, 1],
            ], dtype=np.float64)

            transition_lb = np.array([
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ], dtype=np.float64)

            transition_ub = np.array([
                [1, .2, 0, 0, 0, 0, 0],
                [0.1, 1, .2, 0, 0, 0, 0],
                [0, 0.1, 1, .2, 0, 0, 0],
                [0, 0, 0.1, 1, .2, 0, 0],
                [0, 0, 0, 0.1, 1, .2, 0],
                [0, 0, 0, 0, 0.1, 1, .2],
                [0, 0, 0, 0, 0, 0.1, 1],
            ], dtype=np.float64)
            # print("hahahahahhaha")
            self.init(start_init, transition_init)
            self.set_bounded_start(start_lb, start_ub)
            self.set_bounded_transition(transition_lb, transition_ub)

    #
    # def __dealloc__(self):
    # __dealloc__ 函数子类不会覆盖基类的，两个都会被调用
    # del self.items_info
    # del self.train_items
    # self.items_info = None
    # self.train_items = None
    # 父类已经释放了，基类不能再重复释放，会出问题
    # del self.c_object

    def set_item_info(self, np.ndarray[cython.floating, ndim=2] items):
        """
        设置题目的参数信息，注意行的下标为题目的ID。
        Parameters
        ----------
        items : np.ndarray[double,ndim=2] shape=(n,3) 
            三列分别是题目的slop(区分度)、difficulty(难度)、guess(猜测)

        Returns
        -------

        """

        assert items is not None and items.shape[1] == 3

        # items=items.astype(np.float64)
        if not items.flags['C_CONTIGUOUS']:
            self.items_info = np.ascontiguousarray(items, dtype=np.float64)
        elif items.dtype != np.float64:
            self.items_info = items.astype(dtype=np.float64, order='C')
        else:
            self.items_info = items

        # printf("ptr 1 %x\n",self.items_info)
        # print(items)
        # 注意 c++ 对象的set_items_info 函数保存的传入指针，并没有自己拷贝数据。
        # 所以需要确保传入的指针在对象生命周期不被python自动回收。
        # 这里使用self.items_info持久保存数据，避免被回收
        (<_IRTBKT*> self.c_object).set_items_info(<double *> get_pointer(self.items_info), items.shape[0])

    def set_obs_items(self, np.ndarray[cython.integral, ndim=1] items):
        """
        设置观测序列对应的题目ID，注意这里的题目ID必须是整数，其代表着:set_item_info:函数中传入矩阵的行的下标。
        Parameters
        ----------
        items : np.ndarray[int,ndim=1] shape=(n_x,)
            
        Returns
        -------

        """
        assert items is not None
        # 注意 c++ 对象的 set_items 函数保存的传入指针，并没有自己拷贝数据。
        # 所以需要确保传入的指针在对象生命周期不被python自动回收。
        # 这里使用 self.train_items 持久保存数据，避免被回收
        if not items.flags['C_CONTIGUOUS']:
            self.train_items = np.ascontiguousarray(items, dtype=np.int32)
        elif items.dtype != np.int32:
            self.train_items = items.astype(dtype=np.int32, order='C')
        else:
            self.train_items = items

        (<_IRTBKT*> self.c_object).set_obs_items(<int *> get_pointer(self.train_items), items.shape[0])

    def fit(self, np.ndarray[cython.integral, ndim=1] x,
            np.ndarray[cython.integral, ndim=1] lengths,
            np.ndarray[cython.integral, ndim=1] obs_items=None,
            int max_iter = 20,
            double tol = 1e-2):
        """
        训练模型
        Parameters
        ----------
        x : np.ndarray[int, ndim=1]
            观测序列，训练数据。如果有多个独立的观测序列，首尾衔接的串行在一起。
        lengths : np.ndarray[int, ndim=1]
            每个独立观测序列的长度。
        obs_items : np.ndarray[int, ndim=1]
            每个观测值，对应的题目ID（set_item_info函数中传入矩阵的行的下标）。
        max_iter : int
            默认值20
        tol : double 
            默认值1e-2

        Returns
        -------

        """

        assert x is not None and lengths is not None

        if obs_items is None and self.train_items is None:
            raise ValueError("train_items must not None")

        x = np.ascontiguousarray(x, dtype=np.int32)
        lengths = np.ascontiguousarray(lengths, dtype=np.int32)

        if obs_items is not None:
            self.set_obs_items(obs_items)
        # (<_IRTBKT*>self.c_object).set_obs_items(<int *> get_pointer(train_items),train_items.shape[0])

        return self.c_object.fit(<int*> get_pointer(x), <int*> get_pointer(lengths), lengths.shape[0], max_iter,
                                 tol)

    def predict_next(self, np.ndarray[cython.integral, ndim=1] x, int item_id, str algorithm="viterbi"):
        """
        预测下一个观测值，两种算法viterbi(维特比)和posterior(后验概率分布)。
        Parameters
        ----------
        x : np.ndarray[int, ndim=1]
            观测序列，训练数据。如果有多个独立的观测序列，首尾衔接的串行在一起。
        item_id : int
            待预测的下一个观测值对应的题目ID（set_item_info函数中传入矩阵的行的下标）
        algorithm

        Returns
        -------

        """

        out = np.zeros(self.n_obs, dtype=np.float64, order="C")
        cdef int n_x = 0
        # 预测首次作答结果
        if x is None or x.shape[0] == 0:
            x_arr = None
            n_x = 0
        else:
            n_x = x.shape[0]
            x_arr = np.ascontiguousarray(x, dtype=np.int32)

        if algorithm == 'viterbi':
            (<_IRTBKT*> self.c_object).predict_by_viterbi(<double*> get_pointer(out), <int*> get_pointer(x_arr), n_x,
                                                          item_id)
        elif algorithm == "posterior":
            (<_IRTBKT*> self.c_object).predict_by_posterior(<double*> get_pointer(out), <int*> get_pointer(x_arr), n_x,
                                                            item_id)
        else:
            raise ValueError("Unknown algorithm:%s" % algorithm)

        return out

    def show(self):

        print('success:', self.c_object.success, 'n_stat=%d' % self.n_stat, "n_obs=%d" % self.n_obs)
        print("-" * 10, "start", "-" * 10)
        print(self.start.round(4))
        print("-" * 10, "transition", "-" * 10)
        print(self.transition.round(4))
        # print("-"*10,"emission","-"*10)
        # print(self.emission)
    def debug(self):

        (<_IRTBKT*> self.c_object).debug()

cdef StandardBKT create_standard(_StandardBKT*model):
    cdef StandardBKT result = StandardBKT(1)
    result.c_object = <_HMM*> model
    return result

cdef IRTBKT create_irt(int n_stat, int n_obs, _IRTBKT*model):
    cdef IRTBKT result = IRTBKT(n_stat, 1)
    result.c_object = <_HMM*> model
    return result

cdef class TrainHelper:
    """
    模型训练辅助工具，适合大批量数据训练，提升训练效率。
    """
    cdef _TrainHelper *c_object
    cdef np.ndarray items_info
    cdef int model_type
    cdef int n_stat
    cdef int n_obs
    _results = []

    def __dealloc__(self):
        del self.c_object

    def __cinit__(self, int n_stat=2, int model_type=1):
        # print(n_stat)
        self.c_object = new _TrainHelper(n_stat, 2, model_type)
        # self.n_stat = n_stat
        # self.n_obs = n_obs
        # self.items_info = None

    def __init__(self, int n_stat=2, int model_type=1):
        """

        Parameters
        ----------
        n_stat : int
            隐状态的数量
        model_type : int
            模型的类型，1-标准bkt；2-IRT变种BKT
        """
        self.items_info = None
        self.model_type = model_type
        self.n_stat = n_stat
        self.n_obs = 2
        # self._results = []

    def init(self, np.ndarray[cython.floating, ndim=1] start=None,
             np.ndarray[cython.floating, ndim=2] transition=None,
             np.ndarray[cython.floating, ndim=2] emission=None):
        """
        初始化参数
        Parameters
        ----------
        start : np.ndarray[double,ndim=1]
            初始矩阵
        transition : np.ndarray[double,ndim=2]
            转移矩阵
        emission : np.ndarray[double,ndim=2]
            发射矩阵

        Returns
        -------
            None
        """

        if start is not None:
            assert abs(start.sum() - 1.0) < 1e-12
            start = np.ascontiguousarray(start, dtype=np.float64)

        if transition is not None:
            assert np.all(abs(1.0 - transition.sum(1)) < 1e-12)
            transition = np.ascontiguousarray(transition, dtype=np.float64)

        if emission is not None:
            assert np.all(abs(1.0 - emission.sum(1)) < 1e-12)
            emission = np.ascontiguousarray(emission, dtype=np.float64)

        self.c_object.init(<double*> get_pointer(start), <double*> get_pointer(transition),
                           <double*> get_pointer(emission),False)

    def set_bound_start(self, np.ndarray[cython.floating, ndim=1] lower=None,
                        np.ndarray[cython.floating, ndim=1] upper=None):
        """
        设置初始概率矩阵的约束
        Parameters
        ----------
        lower : np.ndarray[double,ndim=1]
            下限约束
        upper : np.ndarray[double,ndim=1]
            上限约束
        Returns
        -------
            None
        """

        if lower is not None:
            lower = np.ascontiguousarray(lower, dtype=np.float64)
        if upper is not None:
            upper = np.ascontiguousarray(upper, dtype=np.float64)

        self.c_object.set_bound_pi(<double *> get_pointer(lower), <double *> get_pointer(upper),True)

    def set_bound_transition(self, np.ndarray[cython.floating, ndim=2] lower=None,
                             np.ndarray[cython.floating, ndim=2] upper=None):
        """
        设置转移概率矩阵的约束
        Parameters
        ----------
        lower : np.ndarray[double,ndim=2]
            下限约束
        upper : np.ndarray[double,ndim=2]
            上限约束
        Returns
        -------
            None
        """
        if lower is not None:
            lower = np.ascontiguousarray(lower, dtype=np.float64)
        if upper is not None:
            upper = np.ascontiguousarray(upper, dtype=np.float64)
        self.c_object.set_bound_a(<double *> get_pointer(lower), <double *> get_pointer(upper),True)

    def set_bound_emission(self, np.ndarray[cython.floating, ndim=2] lower=None,
                           np.ndarray[cython.floating, ndim=2] upper=None):
        """
        设置发射概率矩阵的约束，IRT变种BKT无需设置。
        Parameters
        ----------
        lower : np.ndarray[double,ndim=2]
            下限约束
        upper : np.ndarray[double,ndim=2]
            上限约束
        Returns
        -------
            None
        """
        if lower is not None:
            lower = np.ascontiguousarray(lower, dtype=np.float64)
        if upper is not None:
            upper = np.ascontiguousarray(upper, dtype=np.float64)

        self.c_object.set_bound_b(<double *> get_pointer(lower), <double *> get_pointer(upper),True)

    def set_item_info(self, np.ndarray[cython.floating, ndim=2] items):
        """
        设置题目的参数信息，仅IRT变种BKT模型适用。
        Parameters
        ----------
        items :np.ndarray[double,ndim=2], shape=(items_count,3)
            三列分别是题目的 slop(区分度)、difficulty(难度)、guess(猜测)

        Returns
        -------

        """
        assert items is not None and items.shape[1] == 3

        if not items.flags['C_CONTIGUOUS']:

            self.items_info = np.ascontiguousarray(items, dtype=np.float64)
        elif items.dtype != np.float64:
            self.items_info = items.astype(dtype=np.float64, order='C')
        else:
            self.items_info = items

        # self.items_info = np.ascontiguousarray(items, dtype=np.float64)
        self.c_object.set_items_info(<double *> get_pointer(self.items_info), items.shape[0],False)

    def fit(self, np.ndarray[cython.integral, ndim=1] trace,
            np.ndarray[cython.integral, ndim=1] group,
            np.ndarray[cython.integral, ndim=1] x,
            np.ndarray[cython.integral, ndim=1] items=None,
            int max_iter = 20,
            double tol = 1e-2):
        """
        开始训练。
        trace 用来区分模型，也就是每个trace ID 会训练一个独立的模型。
        同一个trace id中，不同观测序列用group进行区分，一个group id 代表一个独立的观测序列。

        trace  group  x
        0       0     1
        0       0     0
        0       0     1
        0       1     0
        0       1     1
        0       1     0
        1       0     0
        1       0     0
        1       0     1
        1       1     0
        1       1     0
        1       1     0
        1       1     0
        ...    ...    ...

        trace==0 和trace==1 分别训练两个HMM模型。trace==0中，有group==0，1两个观测序列。


        Parameters
        ----------
        trace : np.ndarray[int32, ndim=1]
            trace ID 序列，和训练数据x一一对应，每个trace id 单独训练一个模型。相同trace id必须是相邻的。
        group : np.ndarray[int32, ndim=1]

        x :  np.ndarray[int32, ndim=1]
            训练数据，观测值序列。
        items : np.ndarray[int32, ndim=1]
            仅IRT变种BKT模型适用,训练数据中，每条数据对应的题目整型ID。
        max_iter : int
            最大迭代次数
        tol : double
            最小收敛阈值

        Returns
        -------

        """
        assert trace is not None
        assert group is not None
        assert x is not None
        trace = np.ascontiguousarray(trace, dtype=np.int32)
        group = np.ascontiguousarray(group, dtype=np.int32)
        x = np.ascontiguousarray(x, dtype=np.int32)

        cdef int length = trace.shape[0]
        if group.shape[0] == length == x.shape[0]:
            pass
        else:
            raise ValueError("trace,group,x长度不一致")
        # 注意，c++的TrainHelper 不会释放模型对象的内存
        self.c_object.fit(<int*> get_pointer(trace),
                          <int*> get_pointer(group),
                          <int*> get_pointer(x),
                          length,
                          <int*> get_pointer(items),
                          max_iter,
                          tol)
        self._update_result()

    @property
    def model_count(self):
        """
        模型的数量
        Returns
        -------

        """
        return self.c_object.model_count
    # cdef _model_dump(self,model):
    cdef void _update_result(self):
        cdef int i
        self._results.clear()
        for i in range(self.c_object.model_count):
            if self.model_type == 1:
                self._results.append(create_standard(<_StandardBKT*> self.c_object.models[i]))
            elif self.model_type == 2:

                self._results.append(create_irt(self.n_stat, self.n_obs, <_IRTBKT*> self.c_object.models[i]))

    @property
    def models(self):
        """
        返回训练好的模型
        Returns
        -------
            list
        """
        return self._results
