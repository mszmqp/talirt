//
// Created by 张振虎 on 2019/1/16.
//

#include "_bkt.h"



void StandardBKT::predict_next(double *out, int *x, int n_x) {
    if (x == NULL) {
        return;
    }
    // fwdlattice[-1]是最后时刻，隐状态的概率分布
//    double ** fwdlattice = this->forward_simple(x,n_x);

    double *buffer = init1D<double>(n_x*this->n_stat);
    MatrixView<double> fwdlattice(n_x, this->n_stat, buffer);

    double ll = this->stat_distributed(buffer,x,n_x);

    double * predict_stat = init1D<double>(this->n_stat);

    for (int k = 0; k < n_obs; ++k) {
        out[k] = 0;
        for (int i = 0; i < n_stat; ++i) {
            // 预测下一时刻隐状态分布
            predict_stat[i] = 0;
            for (int j = 0; j < n_stat; ++j) {
                predict_stat[i] += fwdlattice[n_x - 1][j] * this->A[j][i];
            }

//            out[k] += fwdlattice[n_x % 2][i] * B[i][k];
            out[k] += predict_stat[i] * this->emmit_pdf(i, k, 0);
        }

    }

//    print1D(fwdlattice[n_x % 2], n_obs);

//    if (free_A)free2D(A, n_stat);
//    if (free_B)free2D(B, n_stat);
    free(buffer);
    free(predict_stat);

//    return log_likelihood;

}



double sigmoid(double z) {
    return 1.0 / (1 + exp(-z));
}

double irt(double theta, Item *item,int m = 1) {

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

double IRTBKT::emmit_pdf( int stat, int obs,int x_pos) {
//    std::cerr << "x_pos:" << x_pos <<" stat:"<<stat<<" obs:"<<obs <<std::endl;

    assert(x_pos < this->items_id_length);
    int item_id = this->items_id[x_pos];
    assert(item_id < this->items_length);
    Item *item = this->items+item_id;

//    std::cerr << "item_id:" << item_id<<std::endl;
//    std::cerr << "slop:" << item->slop <<" difficulty:" << item->intercept << " guess:" << item->guess << std::endl;

//    double prob = item->irt(stat);
    double prob = irt(stat,item);
    assert(prob > 0);
    assert(prob < 1);
    return obs ? prob : (1 - prob);
//    return this->B[stat][obs];
}
double IRTBKT::emmit_pdf_ex( int stat, int obs,int item_id) {
//    std::cerr << "x_pos:" << x_pos <<" stat:"<<stat<<" obs:"<<obs <<std::endl;
    assert(item_id < this->items_length);
    Item *item = this->items+item_id;

    double prob = irt(stat,item);
    assert(prob > 0);
    assert(prob < 1);
    return obs ? prob : (1 - prob);
//    return this->B[stat][obs];
}


void IRTBKT::predict_next(double *out, int *x, int n_x,int item_id) {
    if (x == NULL) {
        return;
    }
    // fwdlattice[-1]是最后时刻，隐状态的概率分布
    double *buffer = init1D<double>(n_x*this->n_stat);
    MatrixView<double> fwdlattice(n_x, this->n_stat, buffer);

    double ll = this->stat_distributed(buffer,x,n_x);

    double * predict_stat = init1D<double>(this->n_stat);

    for (int k = 0; k < n_obs; ++k) {
        out[k] = 0;
        for (int i = 0; i < n_stat; ++i) {
            // 预测下一时刻隐状态分布
            predict_stat[i] = 0;
            for (int j = 0; j < n_stat; ++j) {
                predict_stat[i] += fwdlattice[n_x - 1][j] * this->A[j][i];
            }
            out[k] += predict_stat[i] * this->emmit_pdf_ex(i, k, item_id);
        }

    }

//    free2D(fwdlattice, n_x);
    free(predict_stat);
    free(buffer);


}

