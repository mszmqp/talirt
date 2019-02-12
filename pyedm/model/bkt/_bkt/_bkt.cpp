//
// Created by 张振虎 on 2019/1/16.
//

#include "_bkt.h"


double StandardBKT::predict_by_posterior(double *out, int *x, int n_x) {
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


double sigmoid(double z) {
    return 1.0 / (1 + exp(-z));
}

double irt(double theta, Item *item, int m = 1) {

    double z = 0;
    switch (m) {
        case 1:
            z = theta - item->intercept;
            break;
        default:
            z = item->slop * theta - item->intercept;
            break;
    }

    if (m == 3) {
        return sigmoid(z) + item->guess;

    } else {
        return sigmoid(z);
    }

}

IRTBKT::IRTBKT(int n_stat, int n_obs) : HMM(n_stat, 2) {
    this->items = NULL;
    this->items_id = NULL;

}


IRTBKT::~IRTBKT() {

    this->items = NULL;
    this->items_id = NULL;

//    if (this->items) {
//        delete[] this->items;
//    }
//    if (this->items_id != NULL) {
//        delete[] this->items_id;
//    }
}

void IRTBKT::set_items_info(double *items, int length) {

//    assert(getArrayLen<double >(items) == length * 3);

    this->items = (Item *) items;
    this->items_length = length;
}

/*
void IRTBKT::set_items_info(double items_ptr[], int length) {
    if (length <= 0) {
        return;
    }
//    sizeof()
//    assert(getArrayLen(items_ptr) == length * 3);

    if (this->items) {
        delete[] this->items;
    }

    this->items = new Item[length];
    MatrixView<double> item_mt(length, 3, items_ptr);
    for (int i = 0; i < length; ++i) {
        this->items->slop = item_mt[i][0];
        this->items->intercept = item_mt[i][1];
        this->items->guess = item_mt[i][2];
    }
    this->items_length = length;
}
*/

void IRTBKT::set_items(int items_id[], int length) {
//    if (this->items_id != NULL) {
//        delete[] this->items_id;
//    }
//    this->items_id = new int[length];
//    for (int i = 0; i < length; ++i) {
//        this->items_id[i] = items_id[i];
//    }
    this->items_id = items_id;
    this->items_id_length = length;

}

double IRTBKT::emmit_pdf(int stat, int obs, int t) {
//    std::cerr << "x_pos:" << x_pos <<" stat:"<<stat<<" obs:"<<obs <<std::endl;

    assert(t < this->items_id_length);
    int item_id = this->items_id[t];
    assert(item_id < this->items_length);
    Item *item = this->items + item_id;

//    std::cerr << "item_id:" << item_id<<std::endl;
//    std::cerr << "slop:" << item->slop <<" difficulty:" << item->intercept << " guess:" << item->guess << std::endl;

//    double prob = item->irt(stat);
    double prob = irt((double) stat + 0.2, item);
    assert(prob > 0);
    assert(prob < 1);
    prob = obs ? prob : (1 - prob);


//    std::cerr << "emmit_pdf " << " stat:" << stat << " obs:" << obs;
//    std::cerr << " item_id:" << item_id << " slop:" << item->slop << " difficulty:" << item->intercept << " guess:"
//              << item->guess;
//    std::cerr << " prob:" << prob << std::endl;


    return prob;
//    return this->B[stat][obs];
}

double IRTBKT::emmit_pdf_ex(int stat, int obs, int item_id) {
    assert(item_id < this->items_length);
    Item *item = this->items + item_id;


    double prob = irt((double) stat + 0.5, item);
    assert(prob > 0);
    assert(prob < 1);
    prob = obs ? prob : (1 - prob);
//    std::cerr << "emmit_pdf_ex " << " stat:" << stat << " obs:" << obs;
//    std::cerr << " item_id:" << item_id << " slop:" << item->slop << " difficulty:" << item->intercept << " guess:"
//              << item->guess;
//    std::cerr << " prob:" << prob << std::endl;


    return prob;
//    return this->B[stat][obs];
}

void IRTBKT::predict_first(double *out, int item_id) {
    // 没有历史观测序列，相当于预测首次结果

    for (int i = 0; i < this->n_obs; ++i) {
        out[i] = 0;
        for (int j = 0; j < this->n_stat; ++j) {
            out[i] += this->PI[j] * this->emmit_pdf_ex(j, i, item_id);
        }

    }


}

double IRTBKT::predict_by_posterior(double *out, int *x, int n_x, int item_id) {

    if (out == NULL) {
        return 0;
    }

    // 没有历史观测序列，相当于预测首次结果
    if (x == NULL) {
        for (int i = 0; i < this->n_obs; ++i) {
            out[i] = 0;
            for (int j = 0; j < this->n_stat; ++j) {
                out[i] += this->PI[j] * this->emmit_pdf_ex(j, i, item_id);
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
            out[k] += predict_stat[i] * this->emmit_pdf_ex(i, k, item_id);
        }

    }

//    free2D(fwdlattice, n_x);
    free(predict_stat);
    free(buffer);
    return ll;

}

double IRTBKT::predict_by_viterbi(double *out, int *x, int n_x, int item_id) {

//    if (DEBUG) {
//        std::cout << "predict_by_viterbi " << "n_x:" << n_x << " item_id:" << item_id << std::endl;
//    }

    if (out == NULL || x == NULL) {
        return 0;
    }
    int *stat = init1D<int>(n_x);

    double max_prob = this->viterbi(stat, x, n_x);
    for (int i = 0; i < this->n_obs; ++i) {

        out[i] = this->emmit_pdf_ex(stat[n_x - 1], i, item_id);
    }

    free(stat);
    return max_prob;
}