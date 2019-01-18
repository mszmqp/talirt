//
// Created by 张振虎 on 2019/1/16.
//

#include "_bkt.h"


IRTBKT::IRTBKT(int n_stat, int n_obs) : HMM(n_stat, 2) {
    this->items = NULL;
    this->items_id = NULL;

}


IRTBKT::~IRTBKT() {
    if (this->items) {
        delete[] this->items;
    }
//    if (this->items_id != NULL) {
//        delete[] this->items_id;
//    }
}

void IRTBKT::set_items_param(double items_ptr[], int length) {
    if (length <= 0) {
        return;
    }
//    sizeof()
    assert(getArrayLen(items_ptr) == length * 3);

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

double IRTBKT::emmit_pdf(int x_pos, int stat, int obs) {

    assert(x_pos < this->items_id_length);
    int item_id = this->items_id[x_pos];
    assert(item_id < this->items_length);
    Item *item = &this->items[item_id];

    double prob = item->irt(stat);
//    std::cerr << "x_pos " << x_pos << " item_id " << item_id;
//    std::cerr << " potential " << stat <<" difficulty:" << item->intercept << " prob:" << prob << std::endl;
    assert(prob > 0);
    assert(prob < 1);
    return obs ? prob : (1 - prob);
//    return this->B[stat][obs];
}



