# distutils: language = c++
cimport cython
import numpy as np
cimport numpy as np

cdef extern from "_hmm.cpp":
    pass

cdef extern from "_hmm.h":
    cdef cppclass HMM:
        HMM(int, int) except +
        void init(double *pi, double *a, double *b);
        void setBoundedPI(double *lower, double *upper);
        void setBoundedA(double *lower, double *upper);
        void setBoundedB(double *lower, double *upper);
        double estimate(int *x, int *lengths, int n_lengths, int max_iter, double tol);
        double getPI(double *out);
        double getA(double *out);
        double getB(double *out);
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

cdef class pyHMM:
    cdef HMM *c_object
    cdef int n_stat
    cdef int n_obs
    def __cinit__(self, int n_stat=2, int n_obs=2):
        self.c_object = new HMM(n_stat, n_obs)
        self.n_stat = n_stat
        self.n_obs = n_obs

    def init(self, np.ndarray start=None, np.ndarray transition=None, np.ndarray emission=None):
        self.c_object.init(<double*> get_pointer(start), <double*> get_pointer(transition),
                           <double*> get_pointer(emission))

    def set_bounded_start(self, np.ndarray lower=None, np.ndarray upper=None):
        self.c_object.setBoundedPI(<double *> get_pointer(lower), <double *> get_pointer(upper))

    def set_bounded_transition(self, np.ndarray lower=None, np.ndarray upper=None):
        self.c_object.setBoundedA(<double *> get_pointer(lower), <double *> get_pointer(upper))

    def set_bounded_emission(self, np.ndarray lower=None, np.ndarray upper=None):
        self.c_object.setBoundedB(<double *> get_pointer(lower), <double *> get_pointer(upper))

    def estimate(self, np.ndarray[int, ndim=1] x, np.ndarray[int, ndim=1] lengths, int max_iter = 20,
                 double tol = 1e-2):
        return self.c_object.estimate(<int*> get_pointer(x), <int*> get_pointer(lengths), lengths.shape[0], max_iter,
                                      tol)
    def predict_next(self, np.ndarray x, int n_x, np.ndarray start=None, np.ndarray transition=None,
                     np.ndarray emission=None,
                     int n_stat=0, int n_obs=0):
        out = np.zeros(n_obs if n_obs > 0 else self.n_obs, dtype=np.float64)
        self.c_object.predict_next(<double*> get_pointer(out), <int*> get_pointer(x), n_x,
                                   <double*> get_pointer(start),
                                   <double*> get_pointer(transition),
                                   <double*> get_pointer(emission),
                                   n_stat,n_obs)
        return out
        pass
    @property
    def start(self):
        pi = np.zeros(self.n_stat, dtype=np.float64)
        self.c_object.getPI(<double*> get_pointer(pi))
        return pi

    @property
    def transition(self):
        arr = np.zeros(shape=(self.n_stat, self.n_stat), dtype=np.float64)
        self.c_object.getA(<double*> get_pointer(arr))
        return arr

    @property
    def emission(self):
        arr = np.zeros(shape=(self.n_stat, self.n_obs), dtype=np.float64)
        self.c_object.getB(<double*> get_pointer(arr))
        return arr

    @property
    def iter(self):
        return self.c_object.iter

    @property
    def log_likelihood(self):
        return self.c_object.log_likelihood

    def __dealloc__(self):
        del self.c_object
