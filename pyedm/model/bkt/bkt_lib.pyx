# distutils: language = c++
# _distutils: sources = _bkt/_bkt.cpp
# _distutils: include_dirs = _bkt/
cimport cython
import numpy as np
cimport numpy as np

from cython.parallel import prange,parallel
cimport openmp
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
from libcpp cimport bool

# from libcpp cimport new,delete
cdef extern from "_bkt/_hmm.cpp":
    pass
cdef extern from "_bkt/_hmm.h" nogil:
    cdef cppclass _HMM "HMM":
        _HMM(int, int) nogil except +
        void init(double *pi, double *a, double *b) nogil;
        void set_bound_pi(double *lower, double *upper) nogil;
        void set_bound_a(double *lower, double *upper) nogil;
        void set_bound_b(double *lower, double *upper) nogil;
        int estimate(int *x, int *lengths, int n_lengths, int max_iter, double tol) nogil;
        void get_pi(double *out) nogil;
        void get_a(double *out) nogil;
        void get_b(double *out) nogil;
        void predict_by_posterior(double *out, int *x, int n_x) nogil;
        double posterior_distributed(double *out, int *x, int n_x) nogil;
        double viterbi(int *out,int *x,int n_x);
        void predict_by_viterbi(double *out,int *x,int n_x);
        void predict_first(double *out);
        void set_minimum_obs(int value);
        int get_minimum_obs();
        int iter;
        double log_likelihood;
        int n_obs;
        int n_stat;
        int max_iter;
        bool success;

cdef extern from "_bkt/_bkt.cpp":
    pass


cdef extern from "_bkt/_bkt.h" nogil:
    cdef cppclass _StandardBKT "StandardBKT":
        _StandardBKT(int, int) nogil except +
        void init(double *pi, double *a, double *b) nogil;
        void set_bound_pi(double *lower, double *upper) nogil;
        void set_bound_a(double *lower, double *upper) nogil;
        void set_bound_b(double *lower, double *upper) nogil;
        int estimate(int *x, int *lengths, int n_lengths, int max_iter, double tol) nogil;
        void get_pi(double *out) nogil;
        void get_a(double *out) nogil;
        void get_b(double *out) nogil;
        # void predict_next(double *out, int *x, int n_x) nogil;
        double posterior_distributed(double *out, int *x, int n_x) nogil;
        double viterbi(int *out,int *x,int n_x);
        void predict_by_posterior(double *out,int *x,int n_x);
        void predict_by_viterbi(double *out,int *x,int n_x);
        void predict_first(double *out);
        void set_minimum_obs(int value);
        int get_minimum_obs();
        int iter;
        double log_likelihood;
        int n_obs;
        int n_stat;
        int max_iter;
        bool success;

    cdef cppclass _IRTBKT "IRTBKT" :
        _IRTBKT(int, int) nogil except +
        # void set_items_data(double *items, int length) nogil;
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
        void predict_by_posterior(double *out, int *x, int n_x, int item_id) nogil;
        double posterior_distributed(double *out, int *x, int n_x) nogil;
        double viterbi(int *out,int *x,int n_x);
        void predict_by_viterbi(double *out,int *x,int n_x,int item_id);
        void predict_first(double *out);
        void set_minimum_obs(int value);
        int get_minimum_obs();
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
        _TrainHelper(int n_stat , int n_obs , int model_type ) except +
        void init(double *pi , double *a , double *b )
        void set_bound_pi(double *lower , double *upper )
        void set_bound_a(double *lower , double *upper )
        void set_bound_b(double lower[], double upper[] )
        void set_items_info(double items[], int length)
        void run(int trace[], int group[], int x[], int length, int item[], int max_iter,double tol)
        _HMM **models
        int model_count

ctypedef double dtype_t

cdef void *get_pointer(np.ndarray arr):
    if arr is None:
        return NULL
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    # cdef double[::1] lower_view = arr
    return arr.data


# cdef class _Model:



cdef class StandardBKT:
    cdef _HMM *c_object
    cdef int n_stat
    cdef int n_obs
    def __cinit__(self, int no_object=0,*argv,**kwargs):
        if no_object==0:
            self.c_object =<_HMM*> new _StandardBKT(2, 2)
        else:
            self.c_object = NULL

    def __init__(self,int no_object=0):
        self.n_stat = 2
        self.n_obs = 2

        if self.c_object==NULL:
            return

        start_init = np.array([1.0 / self.n_stat] * self.n_stat, dtype=np.float64)
        # assert start_init.sum() == 1
        start_lb = np.array([0] * self.n_stat, dtype=np.float64)
        start_ub = np.array([1] * self.n_stat, dtype=np.float64)

        transition_init = np.array([[0.4, 0.6], [0, 1]])
        transition_lb = np.array([[0, 0], [0, 1]]).astype(np.float64)
        transition_ub = np.array([[1, 1], [0, 1]]).astype(np.float64)
        emission_init = np.array([[0.8, 0.2], [0.2, 0.8]])

        emission_lb = np.array([[0.7, 0], [0, 0.7]]).astype(np.float64)
        emission_ub = np.array([[1, 0.3], [0.3, 1]]).astype(np.float64)
        self.init(start_init,transition_init,emission_init)
        self.set_bounded_start(start_lb,start_ub)
        self.set_bounded_transition(transition_lb,transition_ub)
        self.set_bounded_emission(emission_lb,emission_ub)


    def init(self, np.ndarray[double,ndim=1] start=None,
             np.ndarray[double,ndim=2] transition=None,
             np.ndarray[double,ndim=2] emission=None):
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
        if self.c_object!=NULL:
            self.c_object.init(<double*> get_pointer(start), <double*> get_pointer(transition),
                           <double*> get_pointer(emission))

    def set_bounded_start(self, np.ndarray[double,ndim=1] lower=None, np.ndarray[double,ndim=1] upper=None):
        if self.c_object!=NULL:
            self.c_object.set_bound_pi(<double *> get_pointer(lower), <double *> get_pointer(upper))
        # self.c_object.setBoundedPI(<double *> &lower[0], <double *> &upper[0])

    def set_bounded_transition(self, np.ndarray[double,ndim=2] lower=None, np.ndarray[double,ndim=2] upper=None):
        if self.c_object!=NULL:
            self.c_object.set_bound_a(<double *> get_pointer(lower), <double *> get_pointer(upper))
        # self.c_object.setBoundedA(<double *> &lower[0][0], <double *> &upper[0][0])

    def set_bounded_emission(self, np.ndarray[double,ndim=2] lower=None, np.ndarray[double,ndim=2] upper=None):
        if self.c_object!=NULL:
            self.c_object.set_bound_b(<double *> get_pointer(lower), <double *> get_pointer(upper))
        # self.c_object.setBoundedB(<double *> &lower[0][0], <double *> &lower[0][0])

    def  estimate(self, np.ndarray[int, ndim=1] x, np.ndarray[int, ndim=1] lengths, int max_iter = 20,
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
        if self.c_object==NULL:
            return None
        cdef int * x_ptr = <int*> get_pointer(x)
        cdef int * l_ptr = <int*> get_pointer(lengths)
        cdef int ll = lengths.shape[0]
        cdef int ret = 0
        with nogil:
            ret= self.c_object.estimate(x_ptr,l_ptr, ll, max_iter,tol)

        return ret

    def predict_next(self,int[::1] x=None,str algorithm="viterbi"):

        out = np.zeros(self.n_obs, dtype=np.float64)
        if x is None or x.shape[0] ==0:
            (<_StandardBKT*>self.c_object).predict_first(<double*> get_pointer(out))
            return out

        cdef int n_x = x.shape[0]
        if algorithm=='viterbi':
            if n_x >0:
                (<_StandardBKT*>self.c_object).predict_by_viterbi(<double*> get_pointer(out), <int*> &x[0], n_x)


        elif algorithm == "map":
            (<_StandardBKT*>self.c_object).predict_by_posterior(<double*> get_pointer(out), <int*> &x[0], n_x)
        else:
            raise ValueError("Unknown algorithm:%"%algorithm)
        return out

    def posterior_distributed(self,int[::1] x):

        cdef double ll;
        cdef int n_x = x.shape[0]
        out = np.zeros((n_x,self.n_stat,), dtype=np.float64)

        (<_StandardBKT*>self.c_object).posterior_distributed(<double*> get_pointer(out), <int*> &x[0],n_x)
        return out

    def viterbi(self,int[::1] x):

        cdef double prob;
        cdef int n_x = x.shape[0]
        out = np.zeros(n_x, dtype=np.int32)

        prob = (<_StandardBKT*>self.c_object).viterbi(<int*> get_pointer(out), <int*> &x[0],n_x)

        return out

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

    @property
    def success(self):
        return self.c_object.success

    def __dealloc__(self):
        del self.c_object
    @property
    def minimum_obs(self):
        return self.c_object.get_minimum_obs()

    def set_minimum_obs(self,int value):
        self.c_object.set_minimum_obs(value)

    def show(self):

        print('success:',self.c_object.success,'n_stat=%d'%self.n_stat,"n_obs=%d"%self.n_obs)
        print('n_stat=%d'%self.n_stat,"n_obs=%d"%self.n_obs)
        print("-"*10,"start","-"*10)
        print(self.start.round(4))
        print("-"*10,"transition","-"*10)
        print(self.transition.round(4))
        print("-"*10,"emission","-"*10)
        print(self.emission.round(4))


cdef class IRTBKT(StandardBKT):

    cdef double *items_info

    def __cinit__(self, int n_stat=7, int no_object=0):
    #     # print(n_stat)
        if self.c_object !=NULL:
            del self.c_object
        if no_object == 0:
            self.c_object = <_HMM*> new _IRTBKT(n_stat, 2)
        else:
            self.c_object = NULL


    def __init__(self, int n_stat=7,int no_object=0):
        # super(IRTBKT,self).__init__(n_stat,n_obs)
        self.items_info = NULL
        self.n_stat = n_stat
        self.n_obs = 2

        if self.c_object==NULL:
            return
        start_init = np.array([1.0 / n_stat] * n_stat, dtype=np.float64)
        # assert start_init.sum() == 1
        start_lb = np.array([0] * n_stat, dtype=np.float64)
        start_ub = np.array([1] * n_stat, dtype=np.float64)
        start_ub = np.asarray([0.5,1,1,1,0.5,0.3,0.1])
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
            self.init(start_init,transition_init)
            self.set_bounded_start(start_lb,start_ub)
            self.set_bounded_transition(transition_lb,transition_ub)



    def set_item_info(self,np.ndarray[double,ndim=2] items):
        """
        设置题目的参数信息，
        Parameters
        ----------
        items  shape=(n,3) 三列分别是题目的slop(区分度)、difficulty(难度)、guess(猜测)

        Returns
        -------

        """
        assert items.shape[1] == 3
        items=items.astype(np.float64)

        self.items_info = <double *> get_pointer(items)
        (<_IRTBKT*>self.c_object).set_items_info(self.items_info, items.shape[0])

        pass

    def set_train_items(self,np.ndarray[int,ndim=1] items):

        (<_IRTBKT*>self.c_object).set_items(<int *> get_pointer(items),items.shape[0])

    def estimate(self, np.ndarray[int, ndim=1] x,
                 np.ndarray[int, ndim=1] lengths,
                 np.ndarray[int,ndim=1] train_items=None,
                 int max_iter = 20,
                 double tol = 1e-2):

        if train_items is None:
            raise ValueError("train_items must not None")

        (<_IRTBKT*>self.c_object).set_items(<int *> get_pointer(train_items),train_items.shape[0])

        return self.c_object.estimate(<int*> get_pointer(x), <int*> get_pointer(lengths), lengths.shape[0], max_iter,
                                      tol)


    def predict_next(self,int[::1] x,int item_id,str algorithm="viterbi"):

        cdef int n_x = x.shape[0]
        out = np.zeros(self.n_obs, dtype=np.float64)
        if algorithm=='viterbi':
            (<_IRTBKT*>self.c_object).predict_by_viterbi(<double*> get_pointer(out), <int*> &x[0], n_x,item_id)
        elif algorithm == "map":
            (<_IRTBKT*>self.c_object).predict_by_posterior(<double*> get_pointer(out), <int*> &x[0], n_x,item_id)
        else:
            raise ValueError("Unkonwn algorithm:%" % algorithm)
        return out

    def show(self):

        print('success:',self.c_object.success,'n_stat=%d'%self.n_stat,"n_obs=%d"%self.n_obs)
        print("-"*10,"start","-"*10)
        print(self.start.round(4))
        print("-"*10,"transition","-"*10)
        print(self.transition.round(4))
        # print("-"*10,"emission","-"*10)
        # print(self.emission)

cdef StandardBKT create_standard(_StandardBKT* model):
    cdef StandardBKT result = StandardBKT(1)
    result.c_object = <_HMM*>model
    return result

cdef IRTBKT create_irt(int n_stat, int n_obs, _IRTBKT* model):
    cdef IRTBKT result = IRTBKT(n_stat,1)
    result.c_object = <_HMM*>model
    return result



cdef class TrainHelper:
    cdef _TrainHelper *c_object
    cdef double *items_info
    cdef int model_type
    cdef int n_stat
    cdef int n_obs
    _results = []

    def __cinit__(self, int n_stat=2,int model_type=1):
        # print(n_stat)
        self.c_object = new _TrainHelper(n_stat, 2,model_type)
        # self.n_stat = n_stat
        # self.n_obs = n_obs
        self.items_info = NULL

    def __init__(self, int n_stat=2,int model_type=1):
        self.items_info = NULL
        self.model_type=model_type
        self.n_stat = n_stat
        self.n_obs = 2
        # self._results = []


    def init(self, np.ndarray[double,ndim=1] start=None,
             np.ndarray[double,ndim=2] transition=None,
             np.ndarray[double,ndim=2] emission=None):
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

        self.c_object.init(<double*> get_pointer(start), <double*> get_pointer(transition),<double*> get_pointer(emission))

    def set_bound_start(self, np.ndarray[double,ndim=1] lower=None, np.ndarray[double,ndim=1] upper=None):
        self.c_object.set_bound_pi(<double *> get_pointer(lower), <double *> get_pointer(upper))

    def set_bound_transition(self, np.ndarray[double,ndim=2] lower=None, np.ndarray[double,ndim=2] upper=None):
        self.c_object.set_bound_a(<double *> get_pointer(lower), <double *> get_pointer(upper))

    def set_bound_emission(self, np.ndarray[double,ndim=2] lower=None, np.ndarray[double,ndim=2] upper=None):
        self.c_object.set_bound_b(<double *> get_pointer(lower), <double *> get_pointer(upper))

    def set_item_info(self,np.ndarray[double,ndim=2] items):
        """

        Parameters
        ----------
        items  shape=(n,3) 三列分别是题目的slop(区分度)、difficulty(难度)、guess(猜测)

        Returns
        -------

        """
        self.items_info = <double *> get_pointer(items)
        self.c_object.set_items_info(self.items_info, items.shape[0])


    def run(self, np.ndarray[int, ndim=1] trace,
                 np.ndarray[int, ndim=1] group,
                 np.ndarray[int, ndim=1] x,
                 np.ndarray[int,ndim=1] items=None,
                 int max_iter = 20,
                 double tol = 1e-2):
        cdef int length =trace.shape[0]
        if group.shape[0] == length == x.shape[0]:
            pass
        else:
            raise ValueError("")


        self.c_object.run(<int*> get_pointer(trace),
                        <int*> get_pointer(group),
                        <int*> get_pointer(x),
                        length,
                        <int*> get_pointer(items),
                        max_iter,
                        tol)
        self._update_result()

    @property
    def model_count(self):
        return self.c_object.model_count
    # cdef _model_dump(self,model):
    cdef void _update_result(self):
        cdef int i
        self._results.clear()
        for i in range(self.c_object.model_count):
            if self.model_type==1:
                self._results.append(create_standard(<_StandardBKT*>self.c_object.models[i]))
            elif  self.model_type ==2:

                self._results.append(create_irt(self.n_stat,self.n_obs,<_IRTBKT*>self.c_object.models[i]))

    @property
    def models(self):
        return self._results

cpdef parallel_fit(
            list data,
            int n_stat=2,int n_obs=2,
            np.ndarray start=None,
            np.ndarray transition=None,
            np.ndarray emission=None,
            dict bound={},
            str model = "IRT",
            np.ndarray items_info=None,
            int max_iter = 20,
            double tol = 1e-2,
            int n_jobs=5,
            ):
    if model not in ['IRT','STANDARD']:
        raise ValueError("model must one of ['IRT','STANDARD']")
        # return

    cdef Py_ssize_t i
    cdef int n_task= len(data) # 一共有多少训练任务，一次训练任务可以有多个观测序列
    cdef int **x_ptr = <int**>malloc(n_task * sizeof(int*)) # 每个训练任务的训练数据
    cdef int **length_ptr = <int**>malloc(n_task * sizeof(int*))# 每个训练任务的 各个观测序列长度
    cdef int **item_id_ptr = <int**>malloc(n_task * sizeof(int*)) #  每个训练任务的 训练数据对应的题目id序列
    # np.ndarray[int, ndim=1] x
    # np.ndarray[int, ndim=1] lengths
    every_task_length_ar = np.zeros(n_task,dtype=np.int32)
    # cdef int * every_task_length_ptr =<int*>get_pointer(every_task_length_ar)
    cdef int[::1] every_task_length = every_task_length_ar
    # cdef int [:] hehe
    for i,item in enumerate(data):
        # 所有观测序列
        x_ptr[i] = <int*>get_pointer(item['x'])
        # 每个作答对应的题目ID
        item_id_ptr[i] = <int*>get_pointer(item.get('items_id',None))
        # 每个独立观测序列的长度
        length_ptr[i] = <int*>get_pointer(item['lengths'])
        # 有几个独立观测序列
        every_task_length[i] = item['lengths'].shape[0]
        # hehe = length_ptr[i]
        # print(item['lengths'].shape,item['lengths'],every_task_length[i], length_ptr[i][0],)

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
    cdef int model_type = 0
    # cdef IRTBKT ** objects
    # cdef StandardBKT ** objects_2
    # if model == "IRT":
    cdef _IRTBKT ** objects = <_IRTBKT**>malloc(n_task * sizeof(_IRTBKT*)) # IRTBKT(n_stat,n_obs)[n_task];
    #
    #     model_type = 1
    # elif model == "STANDARD":
    #     objects = <StandardBKT**>malloc(n_task * sizeof(StandardBKT*))
    #     model_type = 2

    for i in range(n_task):
    #     if model_type == 1:
        objects[i] =  new _IRTBKT(n_stat,n_obs)
    #     elif model_type == 2:
    #         objects[i] =  new StandardBKT(n_stat,n_obs)
    # print(n_jobs)
    if True:
    # with nogil, parallel():
    #     openmp.omp_set_num_threads(n_jobs)
        # bkt = NULL # new IRTBKT(n_stat,n_obs)
        for i in range(n_task):
            # bkt = new IRTBKT(n_stat,n_obs)

            objects[i].init(start_ptr,transition_ptr,emission_ptr)
            objects[i].set_bound_pi(start_lb_ptr,start_ub_ptr)
            objects[i].set_bound_a(transition_lb_ptr,transition_ub_ptr)
            objects[i].set_bound_b(emission_lb_ptr,emission_ub_ptr)
            # if model_type == 1:
            objects[i].set_items_info(items_ptr,items_size)
                # objects[i].set_items_info(items_ptr,items_size)
            objects[i].set_items(item_id_ptr[i],every_task_length[i])
            objects[i].estimate(x_ptr[i],length_ptr[i],every_task_length[i],max_iter,tol)
            # del bkt

    free(x_ptr)
    free(length_ptr)
    free(item_id_ptr)
    # print(n_task)
    results = []
    for i in range(n_task):
        bkt = objects[i]
        result={
            'index':i,
            'success':bkt.success,
            # 'trace': trace,
            'n_stat':bkt.n_stat,
            'n_obs':bkt.n_obs,
            'start': np.zeros(n_stat, dtype=np.float64),
            'transition': np.zeros(shape=(n_stat,n_stat), dtype=np.float64),
            'emission': np.zeros(shape=(n_stat,n_obs), dtype=np.float64),
            'log_likelihood': bkt.log_likelihood,

        }
        result.update(data[i])
        bkt.get_pi(<double*>get_pointer(result['start']))
        bkt.get_a(<double*>get_pointer(result['transition']))
        bkt.get_b(<double*>get_pointer(result['emission']))

        results.append(result)
        # showbkt(result)


    for i in range(n_task):
        del objects[i]
    free(objects)

    return results

    # delete objects

cdef showbkt(bkt):
        print(bkt['n_stat'],bkt['n_obs'],bkt['success'])
        print("="*10,"start","="*10)
        print(bkt['start'].round(3))
        print("="*10,"transition","="*10)
        print(bkt['transition'].round(3))
