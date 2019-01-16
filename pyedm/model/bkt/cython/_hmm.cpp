//
// Created by 张振虎 on 2018/10/31.
//

#include "_hmm.h"
//#include <iostream>


HMM::HMM(int n_stat, int n_obs) {
    this->n_obs = n_obs;
    this->n_stat = n_stat;

    this->PI = init1D<double>(this->n_stat);
    this->PI_LOW = init1D<double>(this->n_stat);;
    this->PI_UPPER = init1D<double>(this->n_stat);;
    toZero1D(this->PI, this->n_stat);
    setConstant1D(this->PI_UPPER, this->n_stat, 1.0);

    this->A = init2D<double>(this->n_stat, this->n_stat);
    this->A_LOW = init2D<double>(this->n_stat, this->n_stat);;
    this->A_UPPER = init2D<double>(this->n_stat, this->n_stat);;
    toZero2D(this->A_LOW, this->n_stat, this->n_stat);
    setConstant2D(this->A_UPPER, this->n_stat, this->n_stat, 1.0);

    this->B = init2D<double>(this->n_stat, this->n_obs);
    this->B_LOW = init2D<double>(this->n_stat, this->n_obs);
    this->B_UPPER = init2D<double>(this->n_stat, this->n_obs);

    toZero2D(this->B_LOW, this->n_stat, this->n_obs);
    setConstant2D(this->B_UPPER, this->n_stat, this->n_obs, 1.0);

    this->fwdlattice = NULL;
    this->backlattice = NULL;
    this->x_ptr = NULL;
    this->x_pos = 0;
    this->gammalattice = NULL;
    this->cn = NULL;

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
    bounded1D<double>(this->PI, this->PI_LOW, this->PI_UPPER, this->n_stat);
    bounded2D<double>(this->A, this->A_LOW, this->A_UPPER, this->n_stat, this->n_stat);
    bounded2D<double>(this->B, this->B_LOW, this->B_UPPER, this->n_stat, this->n_obs);
}

void HMM::init(double pi[], double a[], double b[]) {
    if (pi == NULL) {
        for (int i = 0; i < this->n_stat; ++i) {
            this->PI[i] = 1 / this->n_stat;
        }
    } else {
        cpy1D<double>(pi, this->PI, this->n_stat);
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

double HMM::estimate(int *x, int *lengths, int n_lengths, int max_iter, double tol) {

    this->max_iter = max_iter;
    // 最长的序列长度
    int max_n_x = max1D(lengths, n_lengths);

    // 先释放原来的空间
    free2D(this->fwdlattice, max_n_x);
    free2D(this->backlattice, max_n_x);
    free2D(this->gammalattice, max_n_x);
    free(this->cn);

    // 申请新空间
    this->fwdlattice = init2D<double>(max_n_x, this->n_stat);
    this->backlattice = init2D<double>(max_n_x, this->n_stat);
    this->gammalattice = init2D<double>(max_n_x, this->n_stat);
    this->cn = init1D<double>(max_n_x);

    toZero1D(cn, max_n_x);
    toZero2D(this->fwdlattice, max_n_x, this->n_stat);
    toZero2D(this->backlattice, max_n_x, this->n_stat);
    toZero2D(this->gammalattice, max_n_x, this->n_stat);


    double *gamma_sum = init1D<double>(this->n_stat);
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
        toZero1D<double>(gamma_sum, this->n_stat);

        pre_log_likelihood = cur_log_likelihood;
        cur_log_likelihood = 0;

        // 注意gamma_obs_sum gamma_sum xi_sum 是累计了所有观测序列的值
        for (int c = 0; c < n_lengths; ++c) {
//            this->x_pos += n_x; // 当前观测序列的位置
            x_pos += n_x; // 当前序列的位置
            n_x = lengths[c]; // 当前序列的长度
            if (n_x < 2) {
                continue;
            }
            cur_log_likelihood += this->forward(x_pos, n_x, this->PI, this->A);
//            cur_log_likelihood += log_likelihood;
            this->backward(x_pos, n_x, this->PI, this->A);
            // 重制gamma，每个序列独立
//            toZero2D<double>(this->gammalattice, n_x, this->n_stat);

            this->gamma(x_pos, n_x, this->fwdlattice, this->backlattice, gamma_sum);
            this->xi(x_pos, n_x, this->fwdlattice, this->backlattice, xi_sum);


//            cout << "log_likehood " << log_likelihood << endl;
//            cout << "cn" << endl;
//            print1D(this->cn, n_x);
//            cout << "alpha" << endl;
//            print2D(this->fwdlattice, n_x, this->n_stat);
//            printAlpha(this->fwdlattice, this->cn, n_x, this->n_stat);
//            cout << "beta" << endl;
//            print2D(this->backlattice, n_x, this->n_stat);
//            printBeta(this->backlattice, this->cn, n_x, this->n_stat);
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
            for (int t = 0; t < n_x - 1; t++) {
                for (int i = 0; i < this->n_stat; ++i) {
                    gamma_obs_sum[i][x[x_pos + t]] += this->gammalattice[t][i];
                }
            }
        }


        for (int i = 0; i < this->n_stat; ++i) {
            // 计算新的转移概率
            for (int j = 0; j < this->n_stat; ++j) {
                A[i][j] = xi_sum[i][j] / gamma_sum[i];
            }
            // 计算发射概率
            for (int k = 0; k < this->n_obs; ++k) {
                B[i][k] = gamma_obs_sum[i][k] / gamma_sum[i];
            }

        }

        for (int i = 0; i < this->n_stat; ++i) {
            this->PI[i] = PI[i] / n_lengths;
        }
        cpy2D(A, this->A, this->n_stat, this->n_stat);
        cpy2D(B, this->B, this->n_stat, this->n_obs);
        this->bounded();

//        cout << "------" << iter << "-----" << n_lengths << endl;
//        cout << "PI" << endl;
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
    free(gamma_sum);
    free2D(gamma_obs_sum, this->n_stat);
    free2D(xi_sum, this->n_stat);

    free(this->cn);
    free2D(this->fwdlattice, max_n_x);
    free2D(this->backlattice, max_n_x);
    free2D(this->gammalattice, max_n_x);
//    cout << "=================" << endl;
//    cout << "iter" << endl;
//    cout << iter << endl;
//    cout << "log likelihood" << endl;
//    cout << cur_log_likelihood << endl;
    return cur_log_likelihood;

}


/*
 * 缩放因子参考 https://pdfs.semanticscholar.org/4ce1/9ab0e07da9aa10be1c336400c8e4d8fc36c5.pdf
 */
double HMM::forward(int x_pos, int n_x, double *PI, double **A) {


    int *x = this->x_ptr + x_pos;

    double log_likelihood = 0;
    for (int i = 0; i < this->n_stat; i++) {
        this->fwdlattice[0][i] = PI[i] * this->emmit_pdf(x_pos, i, x[0]);
    }
    this->cn[0] = normalize1D(this->fwdlattice[0], this->n_stat);
    log_likelihood += log(this->cn[0]);
    for (int t = 1; t < n_x; t++) {

        for (int i = 0; i < this->n_stat; i++) {
            this->fwdlattice[t][i] = 0;
            for (int j = 0; j < this->n_stat; ++j) {
                this->fwdlattice[t][i] += this->fwdlattice[t - 1][j] * A[j][i];
            }
            this->fwdlattice[t][i] *= this->emmit_pdf(x_pos, i, x[t]);
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
                this->backlattice[t][i] += A[i][j] * this->emmit_pdf(x_pos, j, x[t + 1]) * this->backlattice[t + 1][j];
            }
            this->backlattice[t][i] /= this->cn[t];
        }
    }
    return 0;
}

void HMM::gamma(int x_pos, int n, double **fwdlattice, double **backlattice, double *gamma_sum) {
    int *x = this->x_ptr + x_pos;
    // 为了和xi的长度一样，这里少算一个。
    for (int t = 0; t < n - 1; ++t) {
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

                xi_sum[i][j] += fwdlattice[t][i] * this->A[i][j] * this->emmit_pdf(x_pos, j, x[t + 1]) *
                                backlattice[t + 1][j];
            }
        }

    }

}

double HMM::emmit_pdf(int x_pos, int stat, int obs) {
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
            out[i * this->n_stat + j] = this->B[i][j];
        }
    }
}

void HMM::predict_next(double *out, int *x, int n_x, double *start, double *transition, double *emission, int n_stat,
                       int n_obs) {
    if (x == NULL) {
        return;
    }
    if (n_stat == 0) {
        n_stat = this->n_stat;
    }
    if (n_obs == 0) {
        n_obs = this->n_obs;
    }
    double *PI;
    bool free_A = false, free_B = false;

    if (start == NULL) {
        PI = this->PI;
    } else {
        PI = start;
    }
    if (transition == NULL) {
        double **A = this->A;
    } else {
        MatrixView<double> A(n_stat, n_stat, transition);
    }

    if (emission == NULL) {
        double **B = this->B;
    } else {
        MatrixView<double> B(n_stat, n_obs, emission);
//        cout << "4r4rer "<<B[0][0] <<" "<<B[0][1]<<endl;
//        cout<< B[1][0]<<" " << B[1][1] <<endl;
    }
    // 前向算法推进时，只需要保留两个状态即可，所以申请长度2的序列就行
    double **fwdlattice = init2D<double>(2, n_stat);
    double cn;
    double log_likelihood = 0;

    for (int i = 0; i < n_stat; i++) {
        fwdlattice[0][i] = PI[i] * B[i][x[0]];
    }
    cn = normalize1D(fwdlattice[0], n_stat);
    log_likelihood += log(cn);

    for (int t = 1; t < n_x; t++) {

        for (int i = 0; i < n_stat; i++) {
            // 用t%2 找到 fwdlattice 的位置
            fwdlattice[t % 2][i] = 0;

            for (int j = 0; j < n_stat; ++j) {
                fwdlattice[t % 2][i] += fwdlattice[(t - 1) % 2][j] * A[j][i];
            }
            fwdlattice[t % 2][i] *= B[i][x[t]];
        }
        cn = normalize1D(fwdlattice[t % 2], n_stat);
        log_likelihood += log(cn);
    }
    // fwdlattice[-1]是最后时刻，隐状态的概率分布
    for (int k = 0; k < n_obs; ++k) {
        out[k] = 0;
        for (int i = 0; i < n_stat; ++i) {
            // 预测下一时刻隐状态分布
            fwdlattice[n_x % 2][i] = 0;
            for (int j = 0; j < n_stat; ++j) {
                fwdlattice[n_x % 2][i] += fwdlattice[(n_x - 1) % 2][j] * A[j][i];
            }

            out[k] += fwdlattice[n_x % 2][i] * B[i][k];
        }

    }

//    print1D(fwdlattice[n_x % 2], n_obs);

//    if (free_A)free2D(A, n_stat);
//    if (free_B)free2D(B, n_stat);
    free2D(fwdlattice, 2);
//    return log_likelihood;

}
