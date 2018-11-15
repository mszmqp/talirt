//
// Created by 张振虎 on 2018/11/6.
//

#include "_hmm.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;


int main() {
    double PI[2] = {0.5, 0.5};
    double PI_LOW[2] = {0, 0}, PI_UPPER[] = {1, 1};

    double A[] = {1, 0,
                0.4, 0.6};
//    double *A_ptr[2] = {A[0], A[1]};

    double A_LOW[] = {1, 0, 0, 0};
    double A_UPPER[] = {1, 0, 1, 1};

//    double *A_LOW_ptr[2] = {A_LOW[0],
//                            A_LOW[1]};
//    double *A_UPPER_ptr[2] = {A_UPPER[0],
//                              A_UPPER[1]};

    double B[] = {0.8, 0.2,
                0.2, 0.8};

//    double *B_ptr[2] = {B[0],
//                        B[1]};

    double B_LOW[] = {0.7, 0, 0, 0.7};
    double B_UPPER[] = {1, 0.3, 0.3, 1};

//    double *B_LOW_ptr[2] = {B_LOW[0],
//                            B_LOW[1]};
//    double *B_UPPER_ptr[2] = {B_UPPER[0],
//                              B_UPPER[1]};

    int n_stat = 2, n_obs = 2;

    HMM hmm(2, 2);
    hmm.init(PI, A, B);
    hmm.setBoundedPI(PI_LOW, PI_UPPER);
    hmm.setBoundedA(A_LOW, A_UPPER);
    hmm.setBoundedB(B_LOW, B_UPPER);

    int x = 0, n_x = 0, i = 0, j = 0;
    int *response = new int[50000];
    int *lengths = new int[50000];
    std::string pre_stu, cur_stu, item, kn, line;

    std::ifstream infile;
    infile.open("/Users/zhangzhenhu/Documents/projects/talirt/train_sample.txt");
    while (getline(infile, line)) {
        std::istringstream iss(line);
        iss >> x >> cur_stu >> item >> kn;
        x -= 1;
        response[i] = x;
        i++;
        if (cur_stu != pre_stu && !pre_stu.empty()) {
            lengths[j] = n_x;
            j++;
            n_x = 0;

        }
        n_x += 1;
        pre_stu = cur_stu;


    }
    lengths[j] = n_x;
    j++;

    hmm.estimate(response, lengths, j);


    cout << "PI" << endl;
    print1D(hmm.PI, 2);
    cout << "A" << endl;
    print2D(hmm.A, n_stat, n_stat);
    cout << "B" << endl;
    print2D(hmm.B, n_stat, n_obs);

    cout << "predict next observation" << endl;
    double next[2]={0};
    double start[]={0.955115,0.0444221 };
    double transition[]={1,0,0.0715796,0.92842};
    double emission[]={0.872745 ,0.127255,0.3, 0.7};
    hmm.predict_next(next,response,lengths[0],start,transition,emission);
    print1D(next,2);

    return 0;
}