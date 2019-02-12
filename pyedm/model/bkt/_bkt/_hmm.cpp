//
// Created by 张振虎 on 2018/10/31.
//

#include "_hmm.h"
//#include <iostream>
using namespace std;

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

    this->A = init2D<double>(this->n_stat, this->n_stat);
    this->A_LOW = init2D<double>(this->n_stat, this->n_stat);;
    this->A_UPPER = init2D<double>(this->n_stat, this->n_stat);;
    toZero2D(this->A, this->n_stat, this->n_stat);
    toZero2D(this->A_LOW, this->n_stat, this->n_stat);
    setConstant2D(this->A_UPPER, this->n_stat, this->n_stat, 1.0);

    this->B = init2D<double>(this->n_stat, this->n_obs);
    this->B_LOW = init2D<double>(this->n_stat, this->n_obs);
    this->B_UPPER = init2D<double>(this->n_stat, this->n_obs);
    toZero2D(this->B, this->n_stat, this->n_obs);
    toZero2D(this->B_LOW, this->n_stat, this->n_obs);
    setConstant2D(this->B_UPPER, this->n_stat, this->n_obs, 1.0);

    this->fwdlattice = NULL;
    this->backlattice = NULL;
    this->x_ptr = NULL;
    this->x_pos = 0;
    this->gammalattice = NULL;
    this->cn = NULL;
    this->msg = "Not fit";
    this->minimum_obs = 3;
}

HMM::~HMM() {
    free(this->PI);
    free(this->PI_LOW);
    free(this->PI_UPPER);
    free2D(this->A, this->n_stat);
    free2D(this->A_LOW, this->n_stat);
    free2D(this->A_UPPER, this->n_stat);
    free2D(this->B, this->n_stat);
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
        if (this->B && this->B_LOW && this->B_UPPER) {

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
    // 最长的序列长度
    int max_n_x = max1D(lengths, n_lengths);
    if (max_n_x < this->minimum_obs) {
//        this->success = false;
        this->msg = "The obs length is less than minimum_obs";
//        std::cerr <<"n_lengths:"<<n_lengths<< " max_n_x:" << max_n_x << std::endl;
        return this->success;
    }
//    return true;
    // 先释放原来的空间
//    free2D(this->fwdlattice, max_n_x);
//    free2D(this->backlattice, max_n_x);
//    free2D(this->gammalattice, max_n_x);
//    free(this->cn);
//        print1D<int>(lengths,n_lengths);
//        std::cout<<"--"<<n_lengths<<" "<<max_n_x<<std::endl;
//    return true;
    // 申请新空间
    this->fwdlattice = init2D<double>(max_n_x, this->n_stat);
    this->backlattice = init2D<double>(max_n_x, this->n_stat);
    this->gammalattice = init2D<double>(max_n_x, this->n_stat);
    this->cn = init1D<double>(max_n_x);

    toZero1D(this->cn, max_n_x);
    toZero2D(this->fwdlattice, max_n_x, this->n_stat);
    toZero2D(this->backlattice, max_n_x, this->n_stat);
    toZero2D(this->gammalattice, max_n_x, this->n_stat);


    double *gamma_sum_T = init1D<double>(this->n_stat);
    double *gamma_sum_T_1 = init1D<double>(this->n_stat);

    double **xi_sum = init2D<double>(this->n_stat, this->n_stat);
    double **gamma_obs_sum = init2D<double>(this->n_stat, this->n_obs);


    double *PI = init1D<double>(this->n_stat);
    double **A = init2D<double>(this->n_stat, this->n_stat);
    double **B = init2D<double>(this->n_stat, this->n_obs);


    int n_x, iter;
    int x_pos;
    double cur_log_likelihood = 0, pre_log_likelihood = 0;

    this->x_ptr = x;

    for (iter = 0; iter < max_iter; ++iter) {
//        start_pos = 0;
//        this->x_pos = 0;
        x_pos = 0;
        n_x = 0;
        toZero1D<double>(PI, this->n_stat);
        toZero2D<double>(A, this->n_stat, this->n_stat);
        toZero2D<double>(B, this->n_stat, this->n_obs);

        toZero2D<double>(xi_sum, this->n_stat, this->n_stat);
        toZero2D<double>(gamma_obs_sum, this->n_stat, this->n_obs);
        toZero1D<double>(gamma_sum_T, this->n_stat);
        toZero1D<double>(gamma_sum_T_1, this->n_stat);

        pre_log_likelihood = cur_log_likelihood;
        cur_log_likelihood = 0;

//        std::cout << "------" << iter << "-----" << n_lengths << std::endl;
//        std::cout << "PI" << std::endl;
//        print1D(this->PI, this->n_stat);
//        cout << "A" << endl;
//        print2D(this->A, this->n_stat, this->n_stat);
//        cout << "B" << endl;
//        print2D(this->B, this->n_stat, this->n_obs);


        // 注意gamma_obs_sum gamma_sum xi_sum 是累计了所有观测序列的值
        for (int c = 0; c < n_lengths; ++c) {

//            this->x_pos += n_x; // 当前观测序列的位置
            x_pos += n_x; // 当前序列的位置
            n_x = lengths[c]; // 当前序列的长度
            if (n_x < this->minimum_obs) { // 序列长度至少要是3
                continue;
            }
            cur_log_likelihood += this->forward(x_pos, n_x, this->PI, this->A);
//            cur_log_likelihood += log_likelihood;
            this->backward(x_pos, n_x, this->PI, this->A);
            // 重制gamma，每个序列独立
//            toZero2D<double>(this->gammalattice, n_x, this->n_stat);

            this->gamma(x_pos, n_x, this->fwdlattice, this->backlattice, gamma_sum_T);

            // 计算n_x-1个gamma的和gamma_sum_T_1
            for (int t = 0; t < n_x - 1; ++t) {
                for (int i = 0; i < this->n_stat; ++i) {
                    gamma_sum_T_1[i] += this->gammalattice[t][i];
                }
            }


            this->xi(x_pos, n_x, this->fwdlattice, this->backlattice, xi_sum);


//            std::cout << "log_likehood " << cur_log_likelihood << std::endl;
//            cout << "cn" << endl;
//            print1D(this->cn, n_x);
//            cout << "alpha" << endl;
//            print2D(this->fwdlattice, n_x, this->n_stat);
//          //  --printAlpha(this->fwdlattice, this->cn, n_x, this->n_stat);
//            cout << "beta" << endl;
//            print2D(this->backlattice, n_x, this->n_stat);
//            //--printBeta(this->backlattice, this->cn, n_x, this->n_stat);
//            cout << "gamma" << endl;
//            print2D(this->gammalattice, n_x, this->n_stat);
//            cout << "xi_sum" << endl;
//            print2D(xi_sum, this->n_stat, this->n_stat);

            // 累加每个序列的第一个gamma，用于计算初识概率PI
            for (int i = 0; i < this->n_stat; ++i) {
                PI[i] += this->gammalattice[0][i];
//                this->PI[i]+=PI[i];
            }

//            normalize1D(PI, this->n_stat);

            // 发射概率的分子部分
            for (int t = 0; t < n_x; t++) {
                for (int i = 0; i < this->n_stat; ++i) {
                    gamma_obs_sum[i][x[x_pos + t]] += this->gammalattice[t][i];
                }
            }
        }

//        cout << "xi_sum" << endl;
//        print2D(xi_sum, this->n_stat, this->n_stat);
//        cout << "gamma_sum" << endl;
//        print1D(gamma_sum, this->n_stat);

        for (int i = 0; i < this->n_stat; ++i) {
            // 计算新的转移概率，注意 转移概率的gamma_sum是T-1个求和，
            for (int j = 0; j < this->n_stat; ++j) {
                // 当观测序列都是一个观测值的时候 会出现分母为0
                if (gamma_sum_T_1[i] == 0) {
                    if (xi_sum[i][j] == gamma_sum_T_1[i]) {
                        A[i][j] = 1;
                    }
                } else {

                    A[i][j] = xi_sum[i][j] / gamma_sum_T_1[i];
                }
            }

            // 计算发射概率，注意发射概率的gamma_sum是T个求和
            for (int k = 0; k < this->n_obs; ++k) {
                // 当观测序列都是一个观测值的时候 会出现分母为0
                if (gamma_sum_T[i] == 0) {
                    if (gamma_obs_sum[i][k] == gamma_sum_T[i]) {
                        B[i][k] = 1;
                    } else { // todo 这个分支会出现吗？
                        B[i][k] = 0;
                    }
                } else {
                    B[i][k] = gamma_obs_sum[i][k] / gamma_sum_T[i];

                }
            }

        }

        for (int i = 0; i < this->n_stat; ++i) {
            this->PI[i] = PI[i] / n_lengths;
        }
        cpy2D(A, this->A, this->n_stat, this->n_stat);
        cpy2D(B, this->B, this->n_stat, this->n_obs);
        this->bounded();

//        std::cout << "PI" << std::endl;
//        print1D(this->PI, this->n_stat);
//        cout << "A" << endl;
//        print2D(this->A, this->n_stat, this->n_stat);
//        cout << "B" << endl;
//        print2D(this->B, this->n_stat, this->n_obs);
//        cout << "log likelihood " << cur_log_likelihood << endl;


        if (abs(cur_log_likelihood - pre_log_likelihood) < tol) {
            break;
        }
    }


    this->iter = iter;
    this->log_likelihood = cur_log_likelihood;

    free(PI);
    free2D(A, this->n_stat);
    free2D(B, this->n_stat);
    free(gamma_sum_T_1);
    free(gamma_sum_T);
    free2D(gamma_obs_sum, this->n_stat);
    free2D(xi_sum, this->n_stat);

    free(this->cn);
    free2D(this->fwdlattice, max_n_x);
    free2D(this->backlattice, max_n_x);
    free2D(this->gammalattice, max_n_x);
    this->x_ptr = NULL;
//    cout << "=================" << endl;
//    cout << "iter" << endl;
//    cout << iter << endl;
//    cout << "log likelihood" << endl;
//    cout << cur_log_likelihood << endl;
    this->success = true;
    this->msg = "OK";
    return true;

}


/*
 * 缩放因子参考 https://pdfs.semanticscholar.org/4ce1/9ab0e07da9aa10be1c336400c8e4d8fc36c5.pdf
 */
double HMM::forward(int x_pos, int n_x, double *PI, double **A) {


    int *x = this->x_ptr + x_pos;

    double log_likelihood = 0;
    for (int i = 0; i < this->n_stat; i++) {
        this->fwdlattice[0][i] = PI[i] * this->emmit_pdf(i, x[0], x_pos + 0);
    }
//    if (this->fwdlattice[0][0] == 0) {
//        cout << "--------" << endl;
//        cout << PI[0] << endl;
//        cout << this->emmit_pdf(x_pos + 0, 0, x[0]) << endl;
//    }

    this->cn[0] = normalize1D(this->fwdlattice[0], this->n_stat);
    log_likelihood += log(this->cn[0]);
    for (int t = 1; t < n_x; t++) {

        for (int i = 0; i < this->n_stat; i++) {
            this->fwdlattice[t][i] = 0;
            for (int j = 0; j < this->n_stat; ++j) {
                this->fwdlattice[t][i] += this->fwdlattice[t - 1][j] * A[j][i];
            }
            this->fwdlattice[t][i] *= this->emmit_pdf(i, x[t], x_pos + t);

//            if (this->fwdlattice[t][i] == 0) {
//                cout << "--------" << endl;
//                cout << "PI[i]" << PI[i] << endl;
//                cout << "prob " << this->emmit_pdf(x_pos + t, i, x[t]) << endl;
//            }

        }
        this->cn[t] = normalize1D(this->fwdlattice[t], this->n_stat);
        log_likelihood += log(this->cn[t]);
    }

    return log_likelihood;
}

double HMM::backward(int x_pos, int n_x, double *PI, double **A) {
//    double likelihood = 0;
    int *x = this->x_ptr + x_pos;

    for (int i = 0; i < this->n_stat; ++i) {
        this->backlattice[n_x - 1][i] = 1 / this->cn[n_x - 1];
    }

    for (int t = n_x - 2; t >= 0; --t) {
        for (int i = 0; i < this->n_stat; ++i) {

            this->backlattice[t][i] = 0;
            for (int j = 0; j < this->n_stat; ++j) {
                this->backlattice[t][i] +=
                        A[i][j] * this->emmit_pdf(j, x[t + 1], x_pos + t) * this->backlattice[t + 1][j];
            }
            this->backlattice[t][i] /= this->cn[t];
        }
    }
    return 0;
}

void HMM::gamma(int x_pos, int n, double **fwdlattice, double **backlattice, double *gamma_sum) {
//    int *x = this->x_ptr + x_pos;

    for (int t = 0; t < n; ++t) {
        for (int i = 0; i < this->n_stat; ++i) {
            // 注意这里乘上了this->cn[t]
            // 是因为，在论文中，最后算转移概率和发射概率时，都必须要乘以一下。而不影响算初始概率。
            this->gammalattice[t][i] = fwdlattice[t][i] * backlattice[t][i] * this->cn[t];
            gamma_sum[i] += this->gammalattice[t][i];
        }

    }
}

void HMM::xi(int x_pos, int n, double **fwdlattice, double **backlattice, double **xi_sum) {

    int *x = this->x_ptr + x_pos;
    // xi的值并不需要每个时刻保留，而只用到所有时刻累加的结果。
    // 面对多序列时，全部累加在一起就行
    for (int t = 0; t < n - 1; ++t) {
        for (int i = 0; i < this->n_stat; ++i) {
            for (int j = 0; j < this->n_stat; ++j) {

                xi_sum[i][j] += fwdlattice[t][i] * this->A[i][j] * this->emmit_pdf(j, x[t + 1], x_pos + t) *
                                backlattice[t + 1][j];
            }
        }

    }

}

double HMM::emmit_pdf(int stat, int obs, int t) {
    return this->B[stat][obs];
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
    if (this->A == NULL || out == NULL) {
        return;
    }
    for (int i = 0; i < this->n_stat; ++i) {
        for (int j = 0; j < this->n_stat; ++j) {
            out[i * this->n_stat + j] = this->A[i][j];
        }
    }

}

void HMM::get_b(double *out) {
    if (this->B == NULL || out == NULL) {
        return;
    }
    for (int i = 0; i < this->n_stat; ++i) {
        for (int j = 0; j < this->n_obs; ++j) {
            out[i * this->n_obs + j] = this->B[i][j];
        }
    }
}

double HMM::posterior_distributed(double *out, int *x, int n_x) {


    if (x == NULL || n_x==0) {
        return 0;
    }

    MatrixView<double> posterior(n_x, this->n_stat, out);
//    double cn;
    double log_likelihood = 0;

    // 前向算法
    double **fwdlattice = init2D<double>(n_x, this->n_stat);
    double *cn = init1D<double>(n_x);

    for (int i = 0; i < this->n_stat; i++) {
        fwdlattice[0][i] = this->PI[i] * this->emmit_pdf(i, x[0], 0);
    }
    cn[0] = normalize1D<double>(fwdlattice[0], n_stat);
    log_likelihood += log(cn[0]);

    for (int t = 1; t < n_x; t++) {
        for (int i = 0; i < this->n_stat; i++) {
            fwdlattice[t][i] = 0;
            for (int j = 0; j < this->n_stat; ++j) {
                fwdlattice[t][i] += fwdlattice[t - 1][j] * this->A[j][i];
            }
            fwdlattice[t][i] *= this->emmit_pdf(i, x[t], t);
//            if(isnan(fwdlattice[t][i])){
//                std::cout<<"fwdlattice[t][i] t="<<t << " i=" << i <<" x[t]="<<x[t] ;
//                std::cout<< " emmit_pdf:"<<this->emmit_pdf(i, x[t], t)<<std::endl;
//
//            }
        }
        cn[t] = normalize1D<double>(fwdlattice[t], this->n_stat);

//        if(isnan(cn[t])){
//            std::cout<<"cn[t] t="<<t << " cn[t]=" << cn[t] <<std::endl;
//        }

        log_likelihood += log(cn[t]);
    }

    // 后向算法
    double **backlattice = init2D<double>(n_x, this->n_stat);

    for (int i = 0; i < this->n_stat; ++i) {
        backlattice[n_x - 1][i] = 1 / cn[n_x - 1];
//        backlattice[n_x - 1][i] = 1 ;
//            if(isnan(cn[n_x - 1])){
//                std::cout<<"cn[n_x - 1] "  << cn[n_x - 1] <<std::endl;
//            }
    }

    for (int t = n_x - 2; t >= 0; --t) {
        for (int i = 0; i < this->n_stat; ++i) {
            backlattice[t][i] = 0;
            for (int j = 0; j < this->n_stat; ++j) {
                backlattice[t][i] +=
                        this->A[i][j] * this->emmit_pdf(j, x[t + 1], t) * backlattice[t + 1][j];
            }
            backlattice[t][i] /= cn[t];
        }
    }
    double nor = 0;

    // 后验概率分布,其实就是gamma
    for (int t = 0; t < n_x; ++t) {
        nor = 0;
        for (int i = 0; i < this->n_stat; ++i) {
//            posterior[t][i] = fwdlattice[t][i] * backlattice[t][i] * cn[t] ;
            posterior[t][i] = fwdlattice[t][i] * backlattice[t][i]  ;

//            if(isnan(posterior[t][i])){
//                std::cout<<"fwdlattice[t][i] "  << fwdlattice[t][i] <<std::endl;
//                std::cout<<"backlattice[t][i] "  << backlattice[t][i] <<std::endl;
//            }
            nor += posterior[t][i];
        }

        for (int i = 0; i < this->n_stat; ++i) {
            posterior[t][i] /= nor;
        }
//        cout<<"--t:"<<t <<" ll:"<< nor <<endl;

//        print1D(posterior[t],n_stat);

    }
//    print2D(posterior)
    free(cn);
    free2D(fwdlattice, n_x);
    free2D(backlattice, n_x);
//    cout << "ll:" << log_likelihood<<endl;
    return log_likelihood;
}


double HMM::viterbi(int *out, int *x, int n_x) {


    double **delta = init2D<double>(n_x, this->n_stat);
    int **psi = init2D<int>(n_x, this->n_stat);

//    int t = 0;
    for (int i = 0; i < this->n_stat; ++i) {

        delta[0][i] = this->PI[i] * this->emmit_pdf(i, x[0], 0);
        psi[0][i] = 0;
    }
    // todo 添加对数操作，将乘法改成加法，避免溢出
    double max_prob = -1, tmp = 0;
    int max_stat;
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

            delta[t][i] = max_prob * this->emmit_pdf(max_stat, x[t], t);
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

    return max_prob;
}

double HMM::predict_by_viterbi(double *out, int *x, int n_x) {

//    if(DEBUG){
//        std::cout << "predict_by_viterbi " << std::endl;
//    }
    int *stat = init1D<int>(n_x);
//    double *predict_obs=init1D(this->n_obs);

    double max_prob=this->viterbi(stat, x, n_x);



    for (int i = 0; i < this->n_obs; ++i) {

        out[i] = this->emmit_pdf(stat[n_x - 1], i, n_x);
    }

    free(stat);
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
    if (x == NULL) {
        for (int i = 0; i < this->n_obs; ++i) {
            out[i] = 0;
            for (int j = 0; j < this->n_stat; ++j) {
                out[i] += this->PI[j] * this->emmit_pdf(j, i, 0);
            }

        }

        return 0;
    }

    // fwdlattice[-1]是最后时刻，隐状态的概率分布

    double *buffer = init1D<double>(n_x * this->n_stat);
    MatrixView<double> posterior(n_x, this->n_stat, buffer);

    double ll = this->posterior_distributed(buffer, x, n_x);

    double *predict_stat = init1D<double>(this->n_stat);

    for (int k = 0; k < n_obs; ++k) {
        out[k] = 0;
        for (int i = 0; i < n_stat; ++i) {
            // 预测下一时刻隐状态分布
            predict_stat[i] = 0;
            for (int j = 0; j < n_stat; ++j) {
                predict_stat[i] += posterior[n_x - 1][j] * this->A[j][i];
            }

            out[k] += predict_stat[i] * this->emmit_pdf(i, k, 0);
        }

    }

    free(buffer);
    free(predict_stat);

    return ll;
}
