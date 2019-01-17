//
// Created by 张振虎 on 2019/1/16.
//

#include "_bkt.h"


IRTBKT::IRTBKT(int n_stat, int n_obs) : HMM(n_stat, 2) {
    this->items = NULL;

}

IRTBKT::~IRTBKT() {
    if (this->items) {
        delete[] this->items;
    }
}

void IRTBKT::set_items_param(double items[], int item_size) {
    if (item_size <= 0) {
        return;
    }
//    sizeof()
    assert(getArrayLen(items) == item_size * 3);

    if (this->items) {
        delete[] this->items;
    }

    MatrixView<double> item_mt(item_size, 3, items);
    this->items = new Item[item_size];
    for (int i = 0; i < item_size; ++i) {
        this->items->slop = item_mt[i][0];
        this->items->intercept = item_mt[i][1];
        this->items->guess = item_mt[i][2];
    }

}

void IRTBKT::set_items(int items_id[]) {
    this->items_id = items_id;
}

double IRTBKT::emmit_pdf(int x_pos, int stat, int obs) {

    int item_id = this->items_id[x_pos];
    Item *item = &this->items[item_id];

    double prob = item->irt(stat);
//    std::cerr << "x_pos " << x_pos << " item_id " << item_id;
//    std::cerr << " potential " << stat <<" difficulty:" << item->intercept << " prob:" << prob << std::endl;
    assert(prob>0);
    return obs?prob:(1-prob);
//    return this->B[stat][obs];
}



