//
// Created by 张振虎 on 2018/10/31.
//

#include "hmm.h"
//#include <iostream>
using namespace std;


FitBit::FitBit() {
    this->data = NULL;
    this->data_length = 0;
    this->n_stat = 0;
    this->n_obs = 0;
    this->items = NULL;
    this->fwdlattice = NULL;
    this->backlattice = NULL;
    this->gammalattice = NULL;
    this->cn = NULL;
    this->gamma_sum = NULL;
    this->gamma_sum_less = NULL;

    this->xi_sum = NULL;
    this->gamma_obs_sum = NULL;

//    this->PI_ptr = NULL;
//    this->A_ptr = NULL;
//    this->B_ptr = NULL;
    this->free_data = false;
    this->free_item = false;
    this->free_pi = false;
    this->free_a = false;
    this->free_b = false;

    this->success = false;
    this->log_likelihood = 0;

}

FitBit::~FitBit() {
    if (this->free_data) {
        free(this->data);
    }
    if (this->free_item) {
        free(this->items);
    }
    if (free_pi) {

        free(this->PI);
    }
    if (free_a) {
        free(this->A.data);

    }
    if (free_b) {
        free(this->B.data);
    }


    free2D(this->fwdlattice, this->data_length);
    free2D(this->backlattice, this->data_length);
    free2D(this->gammalattice, this->data_length);

    free(this->cn);
    free(this->gamma_sum);
    free(this->gamma_sum_less);
    free2D(this->xi_sum, this->n_stat);
    free2D(this->gamma_obs_sum, this->n_stat);


}

void FitBit::set_data(int *x, UINT length, bool copy) {

    // 先释放原来的,避免内存泄漏
    if (this->free_data) {
        free(this->data);
    }

    if (copy) {
        this->free_data = true;
        this->data = Calloc(int, length);
        cpy1D<int>(x, this->data, length);
    } else {
        this->free_data = false;
        this->data = x;


    }
    this->data_length = length;
}

void FitBit::set_item(int *item, UINT length, bool copy) {

    // 先释放原来的,避免内存泄漏
    if (this->free_item) {

        free(this->items);
    }

    if (copy) {

        this->free_item = true;
        this->items = Calloc(int, length);
        cpy1D<int>(item, this->items, length);
    } else {
        this->free_item = false;
        this->items = item;


    }
    this->item_length = length;
}


void FitBit::set_pi(double *ptr, bool copy) {
    assert(this->n_stat > 0 && this->n_obs > 0);

    // 先释放原来的,避免内存泄漏
    if (free_pi) {
        free(this->PI);

    }

    if (copy) {
        this->PI = Calloc(double, this->n_stat);

        cpy1D<double>(ptr, this->PI, this->n_stat);
        this->free_pi = true;
//        this->PI.data = this->PI_ptr;
    } else {

        this->free_pi = false;
        this->PI = ptr;
    }
//    this->PI.rows = 1;
//    this->PI.cols = this->n_stat;

}

void FitBit::set_a(double *ptr, bool copy) {
    assert(this->n_stat > 0 && this->n_obs > 0);

    // 先释放原来的,避免内存泄漏
    if (free_a) {
        free(this->A.data);

    }

    if (copy) {
        this->A.data = Calloc(double, this->n_stat * this->n_stat);

        cpy1D<double>(ptr, this->A.data, this->n_stat * this->n_stat);
        this->free_a = true;

    } else {

        this->free_a = false;
        this->A.data = ptr;
    }
    this->A.rows = this->n_stat;
    this->A.cols = this->n_stat;

}

void FitBit::set_b(double *ptr, bool copy) {
    assert(this->n_stat > 0 && this->n_obs > 0);
    // 先释放原来的,避免内存泄漏
    if (free_b) {
        free(this->B.data);

    }

    if (copy) {
        this->B.data = Calloc(double, this->n_stat * this->n_obs);

        cpy1D<double>(ptr, this->B.data, this->n_stat * this->n_obs);
        this->free_b = true;

    } else {
        this->B.data = ptr;
        this->free_b = false;
    }
    this->B.rows = this->n_stat;
    this->B.cols = this->n_obs;

}

void FitBit::init(int n_stat, int n_obs) {

    assert(this->data_length > 0);

    // 先释放原来的空间
    free2D(this->fwdlattice, this->data_length);
    free2D(this->backlattice, this->data_length);
    free2D(this->gammalattice, this->data_length);
    free(this->cn);
    free(this->gamma_sum);
    free(this->gamma_sum_less);
    free2D(this->xi_sum, this->n_stat);
    free2D(this->gamma_obs_sum, this->n_stat);


    this->n_stat = (UINT) n_stat;
    this->n_obs = (UINT) n_obs;
    this->log_likelihood = 0;
    // 申请新空间
    this->fwdlattice = init2D<double>(this->data_length, this->n_stat);
    this->backlattice = init2D<double>(this->data_length, this->n_stat);
    this->gammalattice = init2D<double>(this->data_length, this->n_stat);
    this->cn = init1D<double>(this->data_length);

    this->gamma_sum = init1D<double>(this->n_stat);
    this->gamma_sum_less = init1D<double>(this->n_stat);
    this->xi_sum = init2D<double>(this->n_stat, this->n_stat);
    this->gamma_obs_sum = init2D<double>(this->n_stat, this->n_obs);
}

void FitBit::reset() {

//    toZero1D(this->cn, this->data_length);
//    toZero2D(this->fwdlattice, this->data_length, this->n_stat);
//    toZero2D(this->backlattice, this->data_length, this->n_stat);
//    toZero2D(this->gammalattice, this->data_length, this->n_stat);
    this->log_likelihood = 0;
    this->success = false;
}

FitBit **HMM::covert2fb(int *x, int *lengths, int n_lengths) {
    int n_x = 0;
    int *x_pos = x;

    FitBit **result = Calloc(FitBit*, n_lengths);

    for (int i = 0; i < n_lengths; ++i) {
        // 当前观测序列的起点
        x_pos += n_x;

        // 当前观测序列的长度
        n_x = lengths[i];
        result[i] = new FitBit();
        result[i]->set_data(x_pos, (UINT) n_x);

    }
    return result;
}

FitBit *HMM::covert2fb(int *x, int length) {
    FitBit *fb = new FitBit();
    fb->set_data(x, length);
    return fb;
}
//typedef signed char int;

bool issimplexbounded(double *ar, double *lb, double *ub, int size) {
    double sum = 1;
    for (int i = 0; i < size; i++) {
        sum -= ar[i];
        if (ar[i] < lb[i] || ar[i] > ub[i])
            return false;
    }
//    for(int i=0; i<size && fabs(sum)<SAFETY && fabs(sum)>0; i++) {
//        if( (ar[i]+sum) >=lb[i] ) {
//            ar[i]+=sum;
//            break;
//        }
//        if( (ar[i]+sum) <=ub[i] ) {
//            ar[i]+=sum;
//            break;
//        }
//    }
    return fabs(sum) < SAFETY;
}


void projectsimplexbounded(double *ar, double *lb, double *ub, int size) {
    int i, num_at_hi, num_at_lo; // double of elements at lower,upper boundary
    int *at_hi = Calloc(int, (size_t) size);
    int *at_lo = Calloc(int, (size_t) size);
    double err, lambda, v;
//    double *ar_copy = Calloc(double, (size_t) size);
//    memcpy(ar_copy, ar, (size_t) size);
    int iter = 0;
    for (i = 0; i < size; i++)
        if (ar[i] != ar[i]) {
            fprintf(stderr, "WARNING! NaN detected!\n");
        }
    bool doexit = false;
    while (!issimplexbounded(ar, lb, ub, size) && !doexit) { // 不符合约束条件
        lambda = 0;
        num_at_hi = 0; // 超过上限的元素数量
        num_at_lo = 0; // 超过下限的元素数量
        err = -1;
        // threshold
        for (i = 0; i < size; i++) {
            at_lo[i] = (ar[i] <= lb[i]) ? 1 : 0; // 记录是否低于下限 -- 等于也算？？？？

            v = (at_lo[i] == 1) ? lb[i] : ar[i]; // 如果低于下限，v就是下限值，否则v是原值
            if (v != v) {
                fprintf(stderr, "WARNING! 1 NaN to be set !\n");
            }
            // 如果小于下边界，设置成下边界
            ar[i] = (at_lo[i] == 1) ? lb[i] : ar[i];
            num_at_lo = (int) (num_at_lo + at_lo[i]); // 累加计数器

            at_hi[i] = (ar[i] >= ub[i]) ? 1 : 0;

            v = (at_hi[i] == 1) ? ub[i] : ar[i];
            if (v != v) {
                fprintf(stderr, "WARNING! 2 NaN to be set!\n");
            }
            // 如果大于上边界，设置成上边界
            ar[i] = (at_hi[i] == 1) ? ub[i] : ar[i];
            num_at_hi = (int) (num_at_hi + at_hi[i]);

            err += ar[i]; // ar的和
        }
        // err： 修正后，所有元素的和
        if (err > 0 && size > num_at_lo) // 元素和超过1，并且超过下边界的元素数量 小于 元素总数。(理论上不可能所有元素都超过下边界)
            lambda = err / (size - num_at_lo);
        else if (err < 0 && size > num_at_hi) // 元素和小于1
            lambda = err / (size - num_at_hi);

        int will_suffer_from_lambda = 0; // those values that, if lessened by lambda, will be below 0 or over 1
        double err2 = 0.0, lambda2 = 0.0;
        for (i = 0; i < size; i++) {
            // 如果一个值没有越界
            if ((at_lo[i] == 0 && at_hi[i] == 0)) {
                if (ar[i] < lambda) {
                    will_suffer_from_lambda++;
                    err2 += ar[i];
                }
            }
        }

        if (will_suffer_from_lambda == 0) {
            for (i = 0; i < size; i++) {

                v = ar[i] - ((at_lo[i] == 0 && err > 0) ? lambda : 0);
                if (v != v) {
                    fprintf(stderr, "WARNING! 3 NaN to be set! %f \n", v);
                }

                ar[i] -= (at_lo[i] == 0 && err > 0) ? lambda : 0;

                v = ar[i] - ((at_hi[i] == 0 && err < 0) ? lambda : 0);
                if (v != v) {
                    fprintf(stderr, "WARNING! 4 NaN to be set!\n");
                }

                ar[i] -= (at_hi[i] == 0 && err < 0) ? lambda : 0;
            }
        } else {
            lambda2 = (err - err2) / (size - (num_at_hi + num_at_lo) - will_suffer_from_lambda);
            for (i = 0; i < size; i++) {
                if (at_lo[i] == 0 && at_hi[i] == 0) {

                    v = (ar[i] < lambda) ? 0 : (ar[i] - lambda2);
                    if (v != v) {
                        fprintf(stderr, "WARNING! 5 NaN to be set!\n");
                    }

                    ar[i] = (ar[i] < lambda) ? 0 : (ar[i] - lambda2);
                }
            }
        }
        iter++;
        if (iter == 100) {
            print1D(ar, size);
            print1D(lb, size);
            print1D(ub, size);
            fprintf(stderr, "WARNING! Stuck in projectsimplexbounded().\n");
//            doexit = true;
            exit(1);
        }
    } // until satisfied
    // force last to be 1 - sum of all but last -- this code, actually breaks things
//    err = 0;
//    for(i=0; i<(size); i++) {
//        err += ar[i];
//    }
//    err = 1;
//    for(i=0; i<(size-1); i++) {
//        err -= ar[i];
//    }
//    ar[size-1] = err;
//    err = 1;
//    for(i=1; i<(size-0); i++) {
//        err -= ar[i];
//    }
//    ar[0] = err;

    double sum = 0.0;
    for (i = 0; i < size; i++) {
        sum += ar[i];
        if (ar[i] < 0 || ar[i] > 1)
            fprintf(stderr, "ERROR! projected value is not within [0, 1] range!\n");
    }
//    if( fabs(sum-1)>SAFETY) {
//        fprintf(stderr, "ERROR! projected simplex does not sum to 1!\n");
//    }

    free(at_hi);
    free(at_lo);
}


HMM::HMM(int n_stat, int n_obs) {
    this->success = false;
    this->n_obs = n_obs;
    this->n_stat = n_stat;

    this->PI = init1D<double>(this->n_stat);
    this->PI_LOW = init1D<double>(this->n_stat);;
    this->PI_UPPER = init1D<double>(this->n_stat);;
    toZero1D(this->PI, this->n_stat);
    toZero1D(this->PI_LOW, this->n_stat);
    setConstant1D(this->PI_UPPER, this->n_stat, 1.0);

//    this->A.data = init1D<double>(this->n_stat*this->n_stat);
    this->A.init(this->n_stat, this->n_stat);

    this->A_LOW = init2D<double>(this->n_stat, this->n_stat);;
    this->A_UPPER = init2D<double>(this->n_stat, this->n_stat);;

//    toZero1D(this->A.data, this->n_stat*this->n_stat);
    toZero2D(this->A_LOW, this->n_stat, this->n_stat);
    setConstant2D(this->A_UPPER, this->n_stat, this->n_stat, 1.0);

//    this->B = init2D<double>(this->n_stat, this->n_obs);
    this->B.init(this->n_stat, this->n_obs);

    this->B_LOW = init2D<double>(this->n_stat, this->n_obs);
    this->B_UPPER = init2D<double>(this->n_stat, this->n_obs);
//    toZero2D(this->B, this->n_stat, this->n_obs);
    toZero2D(this->B_LOW, this->n_stat, this->n_obs);
    setConstant2D(this->B_UPPER, this->n_stat, this->n_obs, 1.0);

//    this->fwdlattice = NULL;
//    this->backlattice = NULL;
//    this->x_ptr = NULL;
//    this->x_pos = 0;
//    this->gammalattice = NULL;
//    this->cn = NULL;
    this->msg = "Not fit";
    this->minimum_obs = 3;
}

HMM::~HMM() {
    free(this->PI);
    free(this->PI_LOW);
    free(this->PI_UPPER);

    this->A.free_data();
//    free(this->A.data);

    free2D(this->A_LOW, this->n_stat);
    free2D(this->A_UPPER, this->n_stat);

//    free(this->B.data);
    this->B.free_data();
    free2D(this->B_LOW, this->n_stat);
    free2D(this->B_UPPER, this->n_stat);
}

void HMM::set_bound_pi(double *lower, double *upper) {
    if (lower != NULL) {
        cpy1D(lower, this->PI_LOW, this->n_stat);
    }
    if (upper != NULL) {
        cpy1D(upper, this->PI_UPPER, this->n_stat);
    }
}

void HMM::set_bound_a(double lower[], double upper[]) {

    if (lower != NULL) {
//        gsl_matrix_view m = gsl_matrix_view_array(lower, this->n_stat, this->n_stat);

        MatrixView<double> m(this->n_stat, this->n_stat, lower);
        for (int i = 0; i < this->n_stat; ++i) {
            for (int j = 0; j < this->n_stat; ++j) {
//                this->A_LOW[i][j] = gsl_matrix_get(&m.matrix, i, j);
                this->A_LOW[i][j] = m[i][j];

            }

        }
    }
    if (upper != NULL) {
//        gsl_matrix_view m = gsl_matrix_view_array(upper, this->n_stat, this->n_stat);
        MatrixView<double> m(this->n_stat, this->n_stat, upper);
        for (int i = 0; i < this->n_stat; ++i) {
            for (int j = 0; j < this->n_stat; ++j) {
//                this->A_UPPER[i][j] = gsl_matrix_get(&m.matrix, i, j);
                this->A_UPPER[i][j] = m[i][j];

            }

        }
//        cpy2D(A_UPPER, this->A_UPPER, this->n_stat, this->n_stat);
    }
}

void HMM::set_bound_b(double lower[], double upper[]) {
    if (lower != NULL) {
//        gsl_matrix_view m = gsl_matrix_view_array(lower, this->n_stat, this->n_obs);
        MatrixView<double> m(this->n_stat, this->n_obs, lower);

        for (int i = 0; i < this->n_stat; ++i) {
            for (int j = 0; j < this->n_stat; ++j) {
                this->B_LOW[i][j] = m[i][j];

            }

        }
    }
    if (upper != NULL) {
        MatrixView<double> m(this->n_stat, this->n_obs, upper);

//        gsl_matrix_view m = gsl_matrix_view_array(upper, this->n_stat, this->n_obs);
        for (int i = 0; i < this->n_stat; ++i) {
            for (int j = 0; j < this->n_stat; ++j) {
                this->B_UPPER[i][j] = m[i][j];

            }

        }
    }
}

void HMM::bounded() {
    if (this->PI && this->PI_LOW && this->PI_UPPER) {
//        print1D(this->PI, this->n_stat);
//        print1D(this->PI_LOW, this->n_stat);
//        print1D(this->PI_UPPER, this->n_stat);
        projectsimplexbounded(this->PI, this->PI_LOW, this->PI_UPPER, this->n_stat);

    }
//    bounded1D<double>(this->PI, this->PI_LOW, this->PI_UPPER, this->n_stat);
    for (int i = 0; i < this->n_stat; ++i) {

        projectsimplexbounded(this->A[i], this->A_LOW[i], this->A_UPPER[i], this->n_stat);
        if (this->B.data && this->B_LOW && this->B_UPPER) {

            projectsimplexbounded(this->B[i], this->B_LOW[i], this->B_UPPER[i], this->n_obs);
        }
    }
//    bounded2D<double>(this->A, this->A_LOW, this->A_UPPER, this->n_stat, this->n_stat);
//    bounded2D<double>(this->B, this->B_LOW, this->B_UPPER, this->n_stat, this->n_obs);


}

void HMM::init(double pi[], double a[], double b[]) {
    if (pi == NULL) {
        for (int i = 0; i < this->n_stat; ++i) {
            this->PI[i] = 1 / this->n_stat;
        }
    } else {
//        print1D(pi,this->n_stat);

        cpy1D<double>(pi, this->PI, this->n_stat);
//        print1D(this->PI,this->n_stat);
    }
    if (a == NULL) {
        if (this->n_stat != 2 || this->n_obs != 2) {
            throw "A is NULL";
        } else {
            this->A[0][0] = 1;
            this->A[0][1] = 0;
            this->A[1][0] = 0.4;
            this->A[1][1] = 0.6;
        }


    } else {
        MatrixView<double> m(this->n_stat, this->n_stat, a);
//        gsl_matrix_view m = gsl_matrix_view_array(a, this->n_stat, this->n_stat);
        for (int i = 0; i < this->n_stat; ++i) {
            for (int j = 0; j < this->n_stat; ++j) {
//                this->A[i][j] = gsl_matrix_get(&m.matrix, i, j);
                this->A[i][j] = m[i][j];

            }
        }

//        cpy2D<double>(A, this->A, this->n_stat, this->n_stat);
    }
    if (b == NULL) {

    } else {
//        gsl_matrix_view m = gsl_matrix_view_array(b, this->n_stat, this->n_obs);
        MatrixView<double> m(this->n_stat, this->n_obs, b);
        for (int i = 0; i < this->n_stat; ++i) {
            for (int j = 0; j < this->n_obs; ++j) {
//                this->B[i][j] = gsl_matrix_get(&m.matrix, i, j);
                this->B[i][j] = m[i][j];

            }
        }

    }
}


bool HMM::fit(int *x, int *lengths, int n_lengths, int max_iter, double tol) {

//    std::cout << "n_stat:" << this->n_stat << " n_obs:"<<this->n_obs<<std::endl;
    this->success = false;
    this->max_iter = max_iter;

    FitBit **fb_list = this->covert2fb(x, lengths, n_lengths);
    for (int l = 0; l < n_lengths; ++l) {
        fb_list[l]->init(this->n_stat, this->n_obs);
    }
    FitBit *fb = NULL;


    int iter = 0;
//    int x_pos;
    double cur_log_likelihood = 0, pre_log_likelihood = 0;

    this->log_likelihood_0 = 0;
    this->log_likelihood = 0;


    int valid_count = 0;
    for (iter = 0; iter < this->max_iter; ++iter) {

        pre_log_likelihood = cur_log_likelihood;
        cur_log_likelihood = 0;

//        std::cout << "------" << iter << "-----" << n_lengths << std::endl;
        valid_count = 0;
        for (int c = 0; c < n_lengths; ++c) {
            fb = fb_list[c];

            if (fb->data_length < this->minimum_obs) { // 序列长度至少要是3
                fb->success = false;
                continue;
            }
            valid_count += 1;
            fb->set_pi(this->PI);
            fb->set_a(this->A.data);
            fb->set_b(this->B.data);
            fb->reset();

            this->forward(fb);
            this->backward(fb);
            this->gamma(fb);
            this->xi(fb);
            fb->success = true;
            cur_log_likelihood += fb->log_likelihood;

//            fb->print_alpha();
//            fb->print_beta();
//            fb->print_gamma();
//            fb->print_gamma_sum_less();
//            fb->print_xi_sum();

        } // 所有观测序列前后向算法， 循环结束

        if (iter == 0) {
            this->log_likelihood_0 = cur_log_likelihood;
        }
        // 没有有效的训练数据
        if (valid_count == 0) {
            this->success = false;
            return false;
        }
        this->compute_param(fb_list, n_lengths);

//        cout << "log likelihood " << cur_log_likelihood << endl;
//        cout << "PI" << endl;
//        print1D(this->PI, this->n_stat);
//        cout << "A" << endl;
//        this->A.print();
//        cout << "B" << endl;
//        this->B.print();

        // 检查是否收敛
        if ((cur_log_likelihood - pre_log_likelihood) / valid_count < tol) {
            break;
        }


    }

//    free2D(fb_list,n_lengths);
    for (int i = 0; i < n_lengths; ++i) {
        delete fb_list[i];

    }
    free(fb_list);
    this->iter = iter;
    this->log_likelihood = cur_log_likelihood;

    this->success = true;
    this->msg = "OK";
    return true;

}

double HMM::compute_loglikehood(FitBit **fb_list, int fb_length) {

    FitBit *fb = NULL;
    double log_likelihood = 0;
    for (int c = 0; c < fb_length; ++c) {
        fb = fb_list[c];

        if (fb->data_length < this->minimum_obs) { // 序列长度至少要是3
            fb->success = false;
            continue;
        }
//        valid_count += 1;
        fb->set_pi(this->PI);
        fb->set_a(this->A.data);
        fb->set_b(this->B.data);
        fb->reset();

        this->forward(fb);
        log_likelihood += fb->log_likelihood;
    }
    return log_likelihood;
}

void HMM::compute_param(FitBit **fb_list, int fb_length) {

    double *gamma_sum = init1D<double>(this->n_stat);
    double *gamma_sum_less = init1D<double>(this->n_stat);

    double **xi_sum = init2D<double>(this->n_stat, this->n_stat);
    double **gamma_obs_sum = init2D<double>(this->n_stat, this->n_obs);

    // 计算新的模型参数

    toZero1D<double>(this->PI, this->n_stat);
    this->A.toZero();
    this->B.toZero();
//    toZero2D<double>(A, this->n_stat, this->n_stat);
//    toZero2D<double>(B, this->n_stat, this->n_obs);

    toZero2D<double>(xi_sum, this->n_stat, this->n_stat);
    toZero2D<double>(gamma_obs_sum, this->n_stat, this->n_obs);
    toZero1D<double>(gamma_sum, this->n_stat);
    toZero1D<double>(gamma_sum_less, this->n_stat);
    // 有效的观测序列数量
    int valid_count = 0;
    FitBit *fb = NULL;
    int n_x; // 当前观测序列的观测值长度
    // 累加所有观测序列的一些值
    for (int c = 0; c < fb_length; ++c) {
        fb = fb_list[c];
        n_x = fb->data_length;

        if (!fb->success) { // 当前观测序列 失败了
            continue;
        }
        valid_count += 1;
        // 累加每个序列的第一个gamma，用于计算初识概率PI
        for (int i = 0; i < this->n_stat; ++i) {

            this->PI[i] += fb->gammalattice[0][i];

            gamma_sum[i] += fb->gamma_sum[i];
            gamma_sum_less[i] += fb->gamma_sum_less[i];

            for (int j = 0; j < this->n_stat; ++j) {
                xi_sum[i][j] += fb->xi_sum[i][j];
            }

        }

        // 发射概率的分子部分
        for (int t = 0; t < n_x; t++) { // todo n_x-1 改成 n_x
            for (int i = 0; i < this->n_stat; ++i) {
                gamma_obs_sum[i][fb->data[t]] += fb->gammalattice[t][i];
            }
        }


    } // 累加结束



    for (int i = 0; i < this->n_stat; ++i) {
        // 计算新的转移概率，注意 转移概率的gamma_sum是T-1个求和，
        for (int j = 0; j < this->n_stat; ++j) {
            // 当观测序列都是一个观测值的时候 会出现分母为0
            if (gamma_sum_less[i] == 0) {
                if (xi_sum[i][j] == gamma_sum_less[i]) {
                    this->A[i][j] = 1;
                }
            } else {

                this->A[i][j] = xi_sum[i][j] / gamma_sum_less[i];
            }
        }

        // 计算发射概率，注意发射概率的gamma_sum是T个求和
        for (int k = 0; k < this->n_obs; ++k) {
            // 当观测序列都是一个观测值的时候 会出现分母为0
            if (gamma_sum[i] == 0) {
                if (gamma_obs_sum[i][k] == gamma_sum[i]) { //todo gamma_sum_less 改成 gamma_sum
                    this->B[i][k] = 1;
                } else { // todo 这个分支会出现吗？
                    this->B[i][k] = 0;
                }
            } else {
                this->B[i][k] = gamma_obs_sum[i][k] / gamma_sum[i]; //todo gamma_sum_less 改成 gamma_sum

            }
        }

    }

    for (int i = 0; i < this->n_stat; ++i) {
        this->PI[i] = this->PI[i] / valid_count;
    }


//    cout << "================" << endl;
//    cout << "PI" << endl;
//    print1D(this->PI, 2);
//    cout << "A" << endl;
//    this->A.print();
//    cout << "B" << endl;
//    this->B.print();


    this->bounded();

    free(gamma_sum_less);
    free(gamma_sum);
    free2D(gamma_obs_sum, this->n_stat);
    free2D(xi_sum, this->n_stat);


}

/*
 * 缩放因子参考 https://pdfs.semanticscholar.org/4ce1/9ab0e07da9aa10be1c336400c8e4d8fc36c5.pdf
 */

void HMM::forward(FitBit *fb) {


//    int *x = fb->data;

    double log_likelihood = 0;
    for (int i = 0; i < this->n_stat; i++) {
        fb->fwdlattice[0][i] = fb->PI[i] * this->emmit_pdf(fb, i, 0);
    }


    fb->cn[0] = normalize1D(fb->fwdlattice[0], fb->n_stat);
    log_likelihood += log(1 / fb->cn[0]);

    for (int t = 1; t < fb->data_length; t++) {

        for (int i = 0; i < this->n_stat; i++) {
            fb->fwdlattice[t][i] = 0;
            for (int j = 0; j < this->n_stat; ++j) {
                fb->fwdlattice[t][i] += fb->fwdlattice[t - 1][j] * fb->A[j][i];
            }
            fb->fwdlattice[t][i] *= this->emmit_pdf(fb, i, t);


        }
        fb->cn[t] = normalize1D(fb->fwdlattice[t], this->n_stat);
        log_likelihood += log(1 / fb->cn[t]);
    }
    fb->log_likelihood = log_likelihood;
//    return log_likelihood;
}


void HMM::backward(FitBit *fb) {
//    double likelihood = 0;
//    int *x = fb->data;
    int n_x = fb->data_length;

    for (int i = 0; i < this->n_stat; ++i) {
        fb->backlattice[n_x - 1][i] = 1 / fb->cn[n_x - 1];
    }

    for (int t = n_x - 2; t >= 0; --t) {
        for (int i = 0; i < this->n_stat; ++i) {

            fb->backlattice[t][i] = 0;
            for (int j = 0; j < this->n_stat; ++j) {
                fb->backlattice[t][i] +=
                        fb->A[i][j] * this->emmit_pdf(fb, j, t + 1) * fb->backlattice[t + 1][j];
            }
            fb->backlattice[t][i] /= fb->cn[t];
        }
    }

}


void HMM::gamma(FitBit *fb) {
//    int *x = this->x_ptr + x_pos;
    toZero1D(fb->gamma_sum, fb->n_stat);
    toZero1D(fb->gamma_sum_less, fb->n_stat);

    for (int t = 0; t < fb->data_length; ++t) {
        for (int i = 0; i < fb->n_stat; ++i) {
            // 注意这里乘上了this->cn[t]
            // 是因为，在论文中，最后算转移概率和发射概率时，都必须要乘以一下。而不影响算初始概率。
            fb->gammalattice[t][i] = fb->fwdlattice[t][i] * fb->backlattice[t][i] * fb->cn[t];
            fb->gamma_sum[i] += fb->gammalattice[t][i];
        }

    }
    // gamma_sum_less 比 gamma_sum 少了最后一个
    for (int j = 0; j < fb->n_stat; ++j) {
        fb->gamma_sum_less[j] = fb->gamma_sum[j] - fb->gammalattice[fb->data_length - 1][j];
    }


}


void HMM::xi(FitBit *fb) {

//    int *x = this->x_ptr + x_pos;
//    int *x = fb->data;
    toZero2D<double>(fb->xi_sum, this->n_stat, this->n_stat);
    // xi的值并不需要每个时刻保留，而只用到所有时刻累加的结果。
    // 面对多序列时，全部累加在一起就行
    for (int t = 0; t < fb->data_length - 1; ++t) {
        for (int i = 0; i < fb->n_stat; ++i) {
            for (int j = 0; j < fb->n_stat; ++j) {

                fb->xi_sum[i][j] += fb->fwdlattice[t][i] * fb->A[i][j] * this->emmit_pdf(fb, j, t + 1) *
                                    fb->backlattice[t + 1][j];
            }
        }

    }

}

double HMM::emmit_pdf(int stat, int obs) {
    return this->B[stat][obs];
}

double HMM::emmit_pdf(FitBit *fb, int stat, int t) {
    assert(t < fb->data_length);
    int obs = fb->data[t];
    return fb->B[stat][obs];
}

void HMM::get_pi(double *out) {
    if (this->PI == NULL || out == NULL) {
        return;
    }
    for (int i = 0; i < this->n_stat; ++i) {
        out[i] = this->PI[i];
    }
}

void HMM::get_a(double *out) {
    if (this->A.data == NULL || out == NULL) {
        return;
    }
    for (int i = 0; i < this->n_stat; ++i) {
        for (int j = 0; j < this->n_stat; ++j) {
            out[i * this->n_stat + j] = this->A[i][j];
        }
    }

}

void HMM::get_b(double *out) {
    if (this->B.data == NULL || out == NULL) {
        return;
    }
    for (int i = 0; i < this->n_stat; ++i) {
        for (int j = 0; j < this->n_obs; ++j) {
            out[i * this->n_obs + j] = this->B[i][j];
        }
    }
}

double HMM::posterior_distributed(double *out, int *x, int n_x) {

    FitBit *fb = this->covert2fb(x,n_x);

    double ll = this->posterior_distributed(fb, out);

    delete fb;
    return ll;
}

double HMM::posterior_distributed(FitBit *fb, double *out) {


    if (fb == NULL || fb->data_length == 0 || fb->data == NULL) {
        return 0;
    }

    fb->init(this->n_stat, this->n_obs);

    fb->set_pi(this->PI);
    fb->set_a(this->A.data);
    fb->set_b(this->B.data);

    this->forward(fb);
    this->backward(fb);
    this->gamma(fb);

    // 后验概率分布,其实就是gamma
    MatrixView<double> posterior(fb->data_length, this->n_stat, out);
    for (int t = 0; t < fb->data_length; ++t) {
        for (int i = 0; i < this->n_stat; ++i) {
            posterior[t][i] = fb->gammalattice[t][i];
        }
    }


    return fb->log_likelihood;
}


double HMM::viterbi(int *out, int *x, int n_x) {

    FitBit *fb = covert2fb(x,n_x);

    double **delta = init2D<double>(n_x, this->n_stat);
    int **psi = init2D<int>(n_x, this->n_stat);

    for (int i = 0; i < this->n_stat; ++i) {

//        delta[0][i] = this->PI[i] * this->emmit_pdf(i, x[0], 0);
        delta[0][i] = this->PI[i] * this->emmit_pdf(fb, i, 0);
        psi[0][i] = 0;
    }
    // todo 添加对数操作，将乘法改成加法，避免溢出
    double max_prob = -1, tmp = 0;
    int max_stat = 0;
    for (int t = 1; t < n_x; ++t) {

        for (int i = 0; i < this->n_stat; ++i) {
            max_prob = -1;
            for (int j = 0; j < this->n_stat; ++j) {
                tmp = delta[t - 1][j] * this->A[j][i];
                if (tmp > max_prob) {
                    max_prob = tmp;
                    max_stat = j;
                }
            };

//            delta[t][i] = max_prob * this->emmit_pdf(max_stat, x[t], t);
            delta[t][i] = max_prob * this->emmit_pdf(fb, max_stat, t);
            psi[t][i] = max_stat;
        }
    }

    max_prob = -1;
    max_stat = -1;
    for (int i = 0; i < this->n_stat; ++i) {

        if (delta[n_x - 1][i] > max_prob) {
            max_prob = delta[n_x - 1][i];
            max_stat = i;
        }
    }
    out[n_x - 1] = max_stat;
    double max_tmp;

    for (int t = n_x - 2; t >= 0; --t) {
        max_tmp = -1;
        for (int i = 0; i < this->n_stat; ++i) {
            tmp = delta[t][i] * this->A[i][out[t + 1]];
            if (tmp > max_tmp) {
                max_tmp = tmp;
                out[t] = i;
            }

        }
    }


    free2D(delta, n_x);
    free2D(psi, n_x);
    delete fb;
    return max_prob;
}

double HMM::predict_by_viterbi(double *out, int *x, int n_x) {


    if (out == NULL) {
        return 0;
    }

    int next_stat = -1; // 预测的隐状态
    double _max_t = -1;
    double max_prob = 0; // 隐状态序列的概率

    // 如果没有已知观测序列x，就是预测第一个观测值，基于初始概率PI进行预测
    if (n_x == 0 || x == NULL) {

        // 选出初始概率中最大概率的状态
        for (int i = 0; i < this->n_stat; ++i) {
            if (this->PI[i] > _max_t) {
                _max_t = this->PI[i];
                next_stat = i;
            }
        }

        max_prob = _max_t;

    } else { // 有观测序列x，预测下一个观测值

        int *stat = init1D<int>(n_x);
        // 已知观测序列x，通过viterbi算法解码得到最大概率的隐状态序列
        max_prob = this->viterbi(stat, x, n_x);

        // 预测下一个隐状态的值

        // 当前状态下，最大概率转移到哪一个状态
        for (int j = 0; j < this->n_stat; ++j) {
            if (this->A[stat[n_x - 1]][j] > _max_t) {
                _max_t = this->A[stat[n_x - 1]][j];
                next_stat = j;
            }
        }

        max_prob *= _max_t;
        free(stat);

    }

    assert(next_stat >= 0);

    // 根据预测的下一个隐状态，算出可能的观测值
    for (int i = 0; i < this->n_obs; ++i) {

        out[i] = this->emmit_pdf(next_stat, i);
    }


    return max_prob;
}

void HMM::predict_first(double *out) {
    // 没有历史观测序列，相当于预测首次结果

    for (int i = 0; i < this->n_obs; ++i) {
        out[i] = 0;
        for (int j = 0; j < this->n_stat; ++j) {
            out[i] += this->PI[j] * this->emmit_pdf(j, i);
        }

    }


}

double HMM::predict_by_posterior(double *out, int *x, int n_x) {
    if (out == NULL) {
        return 0;
    }


    // 没有历史观测序列，相当于预测首次结果
    if (x == NULL || n_x == 0) {
        for (int i = 0; i < this->n_obs; ++i) {
            out[i] = 0;
            for (int j = 0; j < this->n_stat; ++j) {
                out[i] += this->PI[j] * this->emmit_pdf(j, i);
            }

        }

        return 0;
    }

//    FitBit *fb = covert2fb(x,n_x);

    // fwdlattice[-1]是最后时刻，隐状态的概率分布

    double *buffer = init1D<double>(n_x * this->n_stat);
    MatrixView<double> posterior(n_x, this->n_stat, buffer);

    double ll = this->posterior_distributed(buffer, x, n_x);
//    double ll = this->posterior_distributed(fb, buffer);

    double *predict_stat = init1D<double>(this->n_stat);

    for (int k = 0; k < this->n_obs; ++k) {
        out[k] = 0;
        for (int i = 0; i < this->n_stat; ++i) {
            // 预测下一时刻隐状态分布
            predict_stat[i] = 0;
            for (int j = 0; j < this->n_stat; ++j) {
                predict_stat[i] += posterior[n_x - 1][j] * this->A[j][i];
            }

            out[k] += predict_stat[i] * this->emmit_pdf(i, k);
        }

    }

    free(buffer);
    free(predict_stat);
//    free(fb);

    return ll;
}
