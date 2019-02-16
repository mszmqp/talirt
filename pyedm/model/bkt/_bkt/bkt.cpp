//
// Created by 张振虎 on 2019/1/16.
//

#include "bkt.h"


//double StandardBKT::predict_by_posterior(double *out, int *x, int n_x) {
//    if (out == NULL) {
//        return 0;
//    }
//
//    // 没有历史观测序列，相当于预测首次结果
//    if (x == NULL) {
//        for (int i = 0; i < this->n_obs; ++i) {
//            out[i] = 0;
//            for (int j = 0; j < this->n_stat; ++j) {
//                out[i] += this->PI[j] * this->emmit_pdf(j, i);
//            }
//
//        }
//
//        return 0;
//    }
//
//    // fwdlattice[-1]是最后时刻，隐状态的概率分布
//
//    double *buffer = init1D<double>(n_x * this->n_stat);
//    MatrixView<double> posterior(n_x, this->n_stat, buffer);
//
//    double ll = this->posterior_distributed(buffer, x, n_x);
//
//    double *predict_stat = init1D<double>(this->n_stat);
//
//    for (int k = 0; k < n_obs; ++k) {
//        out[k] = 0;
//        for (int i = 0; i < n_stat; ++i) {
//            // 预测下一时刻隐状态分布
//            predict_stat[i] = 0;
//            for (int j = 0; j < n_stat; ++j) {
//                predict_stat[i] += posterior[n_x - 1][j] * this->A[j][i];
//            }
//
////            out[k] += predict_stat[i] * this->emmit_pdf(i, k, 0);
//            out[k] += predict_stat[i] * this->emmit_pdf(i, k);
//        }
//
//    }
//
//    free(buffer);
//    free(predict_stat);
//    return ll;
//
//}


double sigmoid(double z) {
    return 1.0 / (1 + exp(-z));
}

double irt(double theta, Item *item, int m = 1) {
//    double D = 1.702;
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

FitBit **IRTBKT::covert2fb(int *x, int *lengths, int n_lengths) {
    assert(this->items_id != NULL);
    FitBit **result = HMM::covert2fb(x, lengths, n_lengths);
    int n_x = 0;

    int *item_pos = this->items_id;

    for (int i = 0; i < n_lengths; ++i) {
        // 当前观测序列的起点
        item_pos += n_x;
        // 当前观测序列的长度
        n_x = lengths[i];
        result[i]->set_item(item_pos, (UINT) n_x);

    }
    return result;
}

FitBit *IRTBKT::covert2fb(int *x, int length) {
    FitBit *fb = new FitBit();
    fb->set_data(x, (UINT) length);
    fb->set_item(this->items_id, (UINT) length);
    return fb;
}

void IRTBKT::set_items_info(double *items, int length) {

//    assert(getArrayLen<double >(items) == length * 3);

    this->items = (Item *) items;
    this->items_length = length;

//    std::cout<< "ptr_2 "<< this->items <<std::endl;
//    print1D<double>((double*)this->items,this->items_length*3);

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

void IRTBKT::set_obs_items(int items_id[], int length) {
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

//double IRTBKT::emmit_pdf(int stat, int obs, int t) {

double IRTBKT::emmit_pdf(FitBit *fb, int stat, int t) {
//    std::cerr << "x_pos:" << x_pos <<" stat:"<<stat<<" obs:"<<obs <<std::endl;

    assert(t < fb->item_length);
    int item_id = fb->items[t];
    assert(item_id < fb->item_length);

//    std::cout << "t:" << t <<" item_id:"<<item_id <<std::endl;

    Item *item = this->items + item_id;
    int obs = fb->data[t];

    double prob = irt((double) stat + 0.2, item);
    assert(prob > 0);
    assert(prob < 1);
    prob = obs ? prob : (1 - prob);

    if (isnan(prob)) {
//        std::cout<< "ptr_2 "<< this->items <<std::endl;
//        std::cout<< "stat=" << stat << " obs="<<obs<<" t="<<t<<std::endl;
//        std::cout<< "item_id=" << item_id << " a="<<item->slop<<" b="<<item->intercept<<" c="<<item->guess<<std::endl;
//        this->debug();
        throw "function:emmit_pdf irt prob is nan!";

    }
    return prob;
//    return this->B[stat][obs];
}

void IRTBKT::debug() {

    std::cout << "stat=" << this->n_stat << " obs=" << this->n_obs << std::endl;
    std::cout << this->items << " ";
    print1D<double>((double *) this->items, this->items_length * 3);

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

    // 没有历史观测序列x，相当于预测首次结果
    if (x == NULL || n_x == 0) {
        for (int i = 0; i < this->n_obs; ++i) {
            out[i] = 0;
            for (int j = 0; j < this->n_stat; ++j) {
                out[i] += this->PI[j] * this->emmit_pdf_ex(j, i, item_id);
            }

        }

        return 0;
    }

//    FitBit *fb = new FitBit();
//    fb->set_data(x, (UINT) n_x);
//    fb->set_item(this->items_id, (UINT) n_x);

    // fwdlattice[-1]是最后时刻，隐状态的概率分布
    double *buffer = init1D<double>(n_x * this->n_stat);

    // 计算隐状态的后验概率分布
    double ll = this->posterior_distributed(buffer, x, n_x);
//    double ll = this->posterior_distributed(fb, buffer);

    MatrixView<double> posterior(n_x, this->n_stat, buffer);


    // 待预测的下一个隐状态概率分布
    double *predict_stat = init1D<double>(this->n_stat);

    for (int k = 0; k < this->n_obs; ++k) {
        out[k] = 0;
        for (int i = 0; i < this->n_stat; ++i) {
            // 预测下一时刻隐状态分布
            predict_stat[i] = 0;
            for (int j = 0; j < this->n_stat; ++j) {
                predict_stat[i] += posterior[n_x - 1][j] * this->A[j][i];
            }
            // 预测观测值
            out[k] += predict_stat[i] * this->emmit_pdf_ex(i, k, item_id);
        }

    }

//    free2D(fwdlattice, n_x);
    free(predict_stat);
    free(buffer);
//    free(fb);
    return ll;

}

double IRTBKT::predict_by_viterbi(double *out, int *x, int n_x, int item_id) {

//    if (DEBUG) {
//        std::cout << "predict_by_viterbi " << "n_x:" << n_x << " item_id:" << item_id << std::endl;
//    }

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

        out[i] = this->emmit_pdf_ex(next_stat, i, item_id);
    }


    return max_prob;
}

//double IRTBKT::posterior_distributed(double *out, int *x, int n_x) {
//
//    FitBit *fb = covert2fb(x, n_x);
//    double ll = HMM::posterior_distributed(fb, out);
//    free(fb);
//    return ll;
//}