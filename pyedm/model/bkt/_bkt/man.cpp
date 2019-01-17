//
// Created by 张振虎 on 2018/11/6.
//

#include "_bkt.h"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

int test_data() {
    int n_stat = 2, n_obs = 2;


    double PI[2] = {0.5, 0.5};
    double PI_LOW[2] = {0, 0}, PI_UPPER[] = {1, 1};

    double A[] = {1, 0,
                  0.4, 0.6};

    double A_LOW[] = {1, 0, 0, 0};
    double A_UPPER[] = {1, 0, 1, 1};

    double B[] = {0.8, 0.2,
                  0.2, 0.8};
    double B_LOW[] = {0.7, 0, 0, 0.7};
    double B_UPPER[] = {1, 0.3, 0.3, 1};
    StandardBKT hmm(2, 2);
    hmm.init(PI, A, B);
    hmm.set_bound_pi(PI_LOW, PI_UPPER);
    hmm.set_bound_a(A_LOW, A_UPPER);
    hmm.set_bound_b(B_LOW, B_UPPER);

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
    double next[2] = {0};
    double start[] = {0.955115, 0.0444221};
    double transition[] = {1, 0, 0.0715796, 0.92842};
    double emission[] = {0.872745, 0.127255, 0.3, 0.7};
    hmm.predict_next(next, response, lengths[0], start, transition, emission);
    print1D(next, 2);

    return 0;
}


int main() {

    test_data();

//    double ar[]={2.97381e-05, 0.0169172, 0.110727, 0.263254, 0.185179, 0.209572, 0.21677};
//    double lb[]={0, 0, 0, 0, 0, 0, 0};
//    double ub[7]={1,1,1,1,1,1,1};
//    print1D(ub,7);
//    projectsimplexbounded(ar,lb,ub,7);






    double PI[2] = {0.5, 0.5};
    double PI_LOW[2] = {0, 0}, PI_UPPER[] = {1, 1};
// 0 会 1不会
//    double A[] = {1, 0, 0.4, 0.6};
//    double A_LOW[] = {1, 0, 0, 0};
//    double A_UPPER[] = {1, 0, 1, 1};
//    double B[] = {0.8, 0.2, 0.2, 0.8};

//    double B_LOW[] = {0.7, 0, 0, 0.7};
//    double B_UPPER[] = {1, 0.3, 0.3, 1};


    // 0 不会，1会 ；0做错，1做对
    double A[] = {0.4, 0.6, 0, 1};
    double A_LOW[] = {0, 0, 0, 1};
    double A_UPPER[] = {1, 1, 0, 1};

    double B[] = {0.8, 0.2, 0.2, 0.8};

    double B_LOW[] = {0.7, 0, 0, 0.7};
    double B_UPPER[] = {1, 0.3, 0.3, 1};

    StandardBKT hmm(2, 2);
    hmm.init(PI, A, B);
    hmm.set_bound_pi(PI_LOW, PI_UPPER);
    hmm.set_bound_a(A_LOW, A_UPPER);
    hmm.set_bound_b(B_LOW, B_UPPER);
//    int x[] = {0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
//    int lengths[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3};
    int lengths[] = {4};
    int x[] = {1, 1, 1, 0};
    hmm.estimate(x, lengths, getArrayLen(lengths));
    return 0;
}
