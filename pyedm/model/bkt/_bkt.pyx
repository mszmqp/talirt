# distutils: language = c++
# _distutils: sources = _bkt/_bkt.cpp
# _distutils: include_dirs = _bkt/
cimport cython
import numpy as np
cimport numpy as np
from cython.parallel import prange,parallel
cimport openmp
from libc.stdlib cimport malloc, free
# from libcpp cimport new,delete
cdef extern from "_bkt/_hmm.cpp":
    pass
cdef extern from "_bkt/_hmm.h":
    pass
cdef extern from "_bkt/_bkt.cpp":
    pass

cdef extern from "_bkt/_bkt.h":
    cdef cppclass StandardBKT:
        StandardBKT(int, int) nogil except +
        void init(double *pi, double *a, double *b) nogil;
        void set_bound_pi(double *lower, double *upper) nogil;
        void set_bound_a(double *lower, double *upper) nogil;
        void set_bound_b(double *lower, double *upper) nogil;
        int estimate(int *x, int *lengths, int n_lengths, int max_iter, double tol) nogil;
        void get_pi(double *out) nogil;
        void get_a(double *out) nogil;
        void get_b(double *out) nogil;
        void predict_next(double *out, int *x, int n_x, double *pi, double *a, double *b, int n_stat, int n_obs) nogil;
        int iter;
        double log_likelihood;
        int n_obs;
        int n_stat;
        int max_iter;

    cdef cppclass IRTBKT:
        IRTBKT(int, int) nogil except +
        void set_items_data(double *items, int length) nogil;
        void set_items_info(double *items, int length) nogil;
        void set_items(int *items_id,int length) nogil;
        void init(double *pi, double *a, double *b) nogil;
        void set_bound_pi(double *lower, double *upper) nogil;
        void set_bound_a(double *lower, double *upper) nogil;
        void set_bound_b(double *lower, double *upper) nogil;
        int estimate(int *x, int *lengths, int n_lengths, int max_iter, double tol) nogil;
        void get_pi(double *out) nogil;
        void get_a(double *out) nogil;
        void get_b(double *out) nogil;
        void predict_next(double *out, int *x, int n_x, double *pi, double *a, double *b, int n_stat, int n_obs) nogil;
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

cdef class _StandardBKT:
    cdef StandardBKT *c_object
    cdef int n_stat
    cdef int n_obs
    def __cinit__(self, int n_stat=2, int n_obs=2):
        self.c_object = new StandardBKT(n_stat, n_obs)
        self.n_stat = n_stat
        self.n_obs = n_obs
    def __init__(self,int n_stat=2, int n_obs=2):
        pass
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
        if start is not None:
            assert abs(start.sum()-1.0) <1e-12
        if transition is not None:
            assert np.all(abs(1.0-transition.sum(1)) <1e-12)
        if emission is not None:
            assert np.all(abs(1.0-emission.sum(1)) <1e-12)

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

    cpdef int estimate(self, np.ndarray[int, ndim=1] x, np.ndarray[int, ndim=1] lengths, int max_iter = 20,
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
        cdef int * x_ptr = <int*> get_pointer(x)
        cdef int * l_ptr = <int*> get_pointer(lengths)
        cdef int ll = lengths.shape[0]
        cdef int ret = 0
        with nogil:
            ret= self.c_object.estimate(x_ptr,l_ptr, ll, max_iter,tol)

        return ret

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

    def show(self):
        print("="*10,"start","="*10)
        print(self.start)
        print("="*10,"transition","="*10)
        print(self.transition)
        print("="*10,"emission","="*10)
        print(self.emission)

cdef class _IRTBKT:
    cdef IRTBKT *c_object
    # cdef int n_stat
    # cdef int n_obs
    cdef double *items_info

    def __cinit__(self, int n_stat=2, int n_obs=2):
        # print(n_stat)
        self.c_object = new IRTBKT(n_stat, n_obs)
        # self.n_stat = n_stat
        # self.n_obs = n_obs
        self.items_info = NULL

    def __init__(self, int n_stat=2, int n_obs=2):
        self.items_info = NULL


    def set_items_param(self,np.ndarray[double,ndim=2] items):
        """

        Parameters
        ----------
        items  shape=(n,3) 三列分别是题目的slop(区分度)、difficulty(难度)、guess(猜测)

        Returns
        -------

        """
        self.items_info = <double *> get_pointer(items)
        self.c_object.set_items_data(self.items_info, items.shape[0])

        pass

    def set_items(self,np.ndarray[int,ndim=1] items):

        self.c_object.set_items(<int *> get_pointer(items),items.shape[0])

    def init(self, np.ndarray[double,ndim=1] start=None, np.ndarray[double,ndim=2] transition=None):
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
        if start is not None:
            assert abs(start.sum()-1.0) <1e-12
        if transition is not None:
            assert np.all(abs(1.0-transition.sum(1)) <1e-12)

        self.c_object.init(<double*> get_pointer(start), <double*> get_pointer(transition),NULL)

    def set_bounded_start(self, np.ndarray[double,ndim=1] lower=None, np.ndarray[double,ndim=1] upper=None):
        self.c_object.set_bound_pi(<double *> get_pointer(lower), <double *> get_pointer(upper))
        # self.c_object.setBoundedPI(<double *> &lower[0], <double *> &upper[0])

    def set_bounded_transition(self, np.ndarray[double,ndim=2] lower=None, np.ndarray[double,ndim=2] upper=None):
        self.c_object.set_bound_a(<double *> get_pointer(lower), <double *> get_pointer(upper))
        # self.c_object.setBoundedA(<double *> &lower[0][0], <double *> &upper[0][0])


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

        # print(self.n_stat, self.n_obs,arr.shape)
        return arr

    @property
    def n_stat(self):
        return self.c_object.n_stat

    @property
    def n_obs(self):
        return self.c_object.n_obs

    @property
    def iter(self):
        return self.c_object.iter

    @property
    def log_likelihood(self):
        return self.c_object.log_likelihood

    def __dealloc__(self):
        del self.c_object

    def show(self):
        print(self.n_stat,self.n_obs)
        print("="*10,"start","="*10)
        print(self.start.round(3))
        print("="*10,"transition","="*10)
        print(self.transition.round(3))
        # print("="*10,"emission","="*10)
        # print(self.emission)

# cdef hehe(int n_stat,int n_obs) nogil:
#
#     cdef IRTBKT bkt = new IRTBKT(n_stat,n_obs)


cpdef batch(
            list data,
            int n_stat=2,int n_obs=2,
            np.ndarray start=None,
            np.ndarray transition=None,
            np.ndarray emission=None,
            dict bound={},
            np.ndarray items_info=None,
            int max_iter = 20,
            double tol = 1e-2
            ):

    cdef Py_ssize_t i
    cdef int **x_ptr = <int**>malloc(len(data) * sizeof(int*))
    cdef int **length_ptr = <int**>malloc(len(data) * sizeof(int*))
    cdef int **item_id_ptr = <int**>malloc(len(data) * sizeof(int*))
    # np.ndarray[int, ndim=1] x
    # np.ndarray[int, ndim=1] lengths
    cdef int n_task= len(data)
    cdef int[::1] every_task_length = np.zeros(n_task,dtype=np.int32)
    for i,item in enumerate(data):
        x_ptr[i] = <int*>get_pointer(item['x'])
        length_ptr[i] = <int*>get_pointer(item['lengths'])
        item_id_ptr[i] = <int*>get_pointer(item.get('items_id',None))
        every_task_length[i] = item['x'].shape[0]

    cdef double *start_ptr = <double*>get_pointer(start)
    cdef double* start_lb_ptr = <double*>get_pointer(bound.get('start_lb',None))
    cdef double* start_ub_ptr = <double*>get_pointer(bound.get('start_ub',None))

    cdef double* transition_ptr = <double*>get_pointer(transition)
    cdef double* transition_lb_ptr = <double*>get_pointer(bound.get('transition_lb',None))
    cdef double* transition_ub_ptr = <double*>get_pointer(bound.get('transition_ub',None))


    cdef double* emission_ptr = <double*>get_pointer(emission)
    cdef double* emission_lb_ptr = <double*>get_pointer(bound.get('emission_lb',None))
    cdef double* emission_ub_ptr = <double*>get_pointer(bound.get('emission_ub',None))

    cdef double*  items_ptr = <double*>get_pointer(items_info)
    cdef int items_size = items_info.shape[0]
    # cdef IRTBKT * objects = NULL #new[n_task] IRTBKT;
    cdef IRTBKT ** objects = <IRTBKT**>malloc(n_task * sizeof(IRTBKT*)) # IRTBKT(n_stat,n_obs)[n_task];
    cdef IRTBKT *bkt;
    for i in range(n_task):
        objects[i] =  new IRTBKT(n_stat,n_obs)

    with nogil, parallel():
        openmp.omp_set_num_threads(10)
        # bkt = NULL # new IRTBKT(n_stat,n_obs)

        for i in prange(n_task):
            # bkt = new IRTBKT(n_stat,n_obs)
            x = x_ptr[i]

            length = length_ptr[i]

            objects[i].init(start_ptr,transition_ptr,emission_ptr)
            objects[i].set_bound_pi(start_lb_ptr,start_ub_ptr)
            objects[i].set_bound_a(transition_lb_ptr,transition_ub_ptr)
            objects[i].set_bound_b(emission_lb_ptr,emission_ub_ptr)
            objects[i].set_items_data(items_ptr,items_size)
                # objects[i].set_items_info(items_ptr,items_size)
            objects[i].set_items(item_id_ptr[i],every_task_length[i])
            # objects[i].estimate(x_ptr[i],length_ptr[i],every_task_length[i],max_iter,tol)
            # del bkt

    free(x_ptr)
    free(length_ptr)
    free(item_id_ptr)
    print(n_task)
    results = []
    for i in range(n_task):
        bkt = objects[i]
        result={
            # 'trace': trace,
            'n_stat':bkt.n_stat,
            'n_obs':bkt.n_obs,
            'start': np.zeros(n_stat, dtype=np.float64),
            'transition': np.zeros(shape=(n_stat,n_stat), dtype=np.float64),
            'emission': np.zeros(shape=(n_stat,n_obs), dtype=np.float64),
            'log_likelihood': bkt.log_likelihood,

        }
        bkt.get_pi(<double*>get_pointer(result['start']))
        bkt.get_a(<double*>get_pointer(result['transition']))
        bkt.get_b(<double*>get_pointer(result['emission']))

        results.append(result)
        showbkt(result)


    for i in range(n_task):
        del objects[i]
    free(objects)

    return results

    # delete objects

cdef showbkt(bkt):
        print(bkt['n_stat'],bkt['n_obs'])
        print("="*10,"start","="*10)
        print(bkt['start'].round(3))
        print("="*10,"transition","="*10)
        print(bkt['transition'].round(3))
