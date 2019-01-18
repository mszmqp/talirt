//
// Created by 张振虎 on 2019/1/16.
//

#ifndef HMM_UTILS_H
#define HMM_UTILS_H

#include <iostream>
//using namespace std;

#define Malloc(type, n) (type *)malloc((n)*sizeof(type))
#define Calloc(type, n) (type *)calloc(n,sizeof(type))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define SAFETY 1e-12 // value to substitute for zero for safe math

template<typename T>
void print2D(T **ar, int size1, int size2) {
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++)
            std::cout << ar[i][j] << " ";
        std::cout << std::endl;
    }

}

template<typename T>
void print1D(T *ar, int size1) {
    for (int i = 0; i < size1; i++) {
        std::cout << ar[i] << " ";
    }
    std::cout << std::endl;

}

template<typename T>
void toZero1D(T *ar, int size) {
    for (int i = 0; i < size; i++)
        ar[i] = 0;
}

template<typename T>
void setConstant1D(T *ar, int size, T value) {
    for (int i = 0; i < size; i++)
        ar[i] = value;
}

template<typename T>
void toZero2D(T **ar, int size1, int size2) {
    for (int i = 0; i < size1; i++)
        for (int j = 0; j < size2; j++)
            ar[i][j] = 0;
}

template<typename T>
void setConstant2D(T **ar, int size1, int size2, T value) {
    for (int i = 0; i < size1; i++)
        for (int j = 0; j < size2; j++)
            ar[i][j] = value;
}


template<typename T>
void toZero3D(T ***ar, int size1, int size2, int size3) {
    for (int i = 0; i < size1; i++)
        for (int j = 0; j < size2; j++)
            for (int l = 0; l < size3; l++)
                ar[i][j][l] = 0;
}

template<typename T>
void toZero4D(T ****ar, int size1, int size2, int size3, int size4) {
    for (int i = 0; i < size1; i++)
        for (int j = 0; j < size2; j++)
            for (int l = 0; l < size3; l++)
                for (int n = 0; n < size4; n++)
                    ar[i][j][l][n] = 0;
}

template<typename T>
T *init1D(int size) {
    T *ar = Calloc(T, (size_t) size);
//    toZero1D()
    return ar;
}

template<typename T>
T **init2D(int size1, int size2) {
    T **ar = (T **) Calloc(T *, (size_t) size1);
    for (int i = 0; i < size1; i++)
        ar[i] = (T *) Calloc(T, (size_t) size2);
    return ar;
}

template<typename T>
T ***init3D(int size1, int size2, int size3) {
    int i, j;
    T ***ar = Calloc(T **, (size_t) size1);
    for (i = 0; i < size1; i++) {
        ar[i] = Calloc(T*, (size_t) size2);
        for (j = 0; j < size2; j++)
            ar[i][j] = Calloc(T, (size_t) size3);
    }
    return ar;
}

template<typename T>
T ****init4D(int size1, int size2, int size3, int size4) {
    int i, j, l;
    T ****ar = Calloc(T ***, (size_t) size1);
    for (i = 0; i < size1; i++) {
        ar[i] = Calloc(T**, (size_t) size2);
        for (j = 0; j < size2; j++) {
            ar[i][j] = Calloc(T*, (size_t) size3);
            for (l = 0; l < size3; l++)
                ar[i][j][l] = Calloc(T, (size_t) size4);
        }
    }
    return ar;
}


template<typename T>
void free2D(T **ar, int size1) {
    if (ar == NULL) {
        return;
    }
    for (int i = 0; i < size1; i++) {
        if (ar[i] != NULL)
            free(ar[i]);
    }
    free(ar);
    //    &ar = NULL;
}

template<typename T>
void free3D(T ***ar, int size1, int size2) {
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++)
            free(ar[i][j]);
        free(ar[i]);
    }
    free(ar);
    //    &ar = NULL;
}

template<typename T>
void free4D(T ****ar, int size1, int size2, int size3) {
    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            for (int l = 0; l < size3; l++)
                free(ar[i][j][l]);
            free(ar[i][j]);
        }
        free(ar[i]);
    }
    free(ar);
    //    &ar = NULL;
}


template<typename T>
void cpy1D(T *source, T *target, int size) {
    memcpy(target, source, sizeof(T) * (size_t) size);
}

template<typename T>
void cpy2D(T **source, T **target, int size1, int size2) {
    for (int i = 0; i < size1; i++)
        memcpy(target[i], source[i], sizeof(T) * (size_t) size2);
}

template<typename T>
void cpy3D(T ***source, T ***target, int size1, int size2, int size3) {
    for (int t = 0; t < size1; t++)
        for (int i = 0; i < size2; i++)
            memcpy(target[t][i], source[t][i], sizeof(T) * (size_t) size3);
}

template<typename T>
void cpy4D(T ****source, T ****target, int size1, int size2, int size3, int size4) {
    for (int t = 0; t < size1; t++)
        for (int i = 0; i < size2; i++)
            for (int j = 0; j < size3; j++)
                memcpy(target[t][i][j], source[t][i][j], sizeof(T) * (size_t) size4);
}


template<typename T>
void swap1D(T *source, T *target, int size) {
    T *buffer = init1D<T>(size); // init1<NUMBER>(size);
    memcpy(buffer, target, sizeof(T) * (size_t) size); // reversed order, destination then source
    memcpy(target, source, sizeof(T) * (size_t) size);
    memcpy(source, buffer, sizeof(T) * (size_t) size);
    free(buffer);
}

template<typename T>
void swap2D(T **source, T **target, int size1, int size2) {
    T **buffer = init2D<T>(size1, size2);
    cpy2D<T>(buffer, target, size1, size2);
    cpy2D<T>(target, source, size1, size2);
    cpy2D<T>(source, buffer, size1, size2);
    free2D<T>(buffer, size1);
}

template<typename T>
void swap3D(T ***source, T ***target, int size1, int size2, int size3) {
    T ***buffer = init3D<T>(size1, size2, size3);
    cpy3D<T>(buffer, target, size1, size2, size3);
    cpy3D<T>(target, source, size1, size2, size3);
    cpy3D<T>(source, buffer, size1, size2, size3);
    free3D<T>(buffer, size1, size2);
}

template<typename T>
void swap4D(T ****source, T ****target, int size1, int size2, int size3, int size4) {
    T ****buffer = init4D<T>(size1, size2, size3, size4);
    cpy4D<T>(buffer, target, size1, size2, size3, size4);
    cpy4D<T>(target, source, size1, size2, size3, size4);
    cpy4D<T>(source, buffer, size1, size2, size3, size4);
    free4D<T>(buffer, size1, size2, size3);
}

template<typename T>
T max1D(T *ar, int size1) {
    T value = 0;
    if (size1 < 1) {
        return value;
    }
    value = ar[0];
    for (int i = 0; i < size1; ++i) {
        if (ar[i] > value) {
            value = ar[i];
        }
    }
    return value;
}


template<typename T>
T sum1D(T *ar, int size1) {
    T sum = 0;

    for (int i = 0; i < size1; ++i) {
        sum += ar[i];
    }
    return sum;
}

template<typename T>
T sum2D(T **ar, int size1, int size2) {
    T sum = 0;

    for (int i = 0; i < size1; ++i) {
        for (int j = 0; j < size2; ++j) {
            sum += ar[i][j];
        }
    }
    return sum;
}

template<typename T>
double normalize1D(T *ar, int size1) {
    T sum = sum1D(ar, size1);
    for (int i = 0; i < size1; ++i) {
        ar[i] /= sum;
    }
    return sum;
}

template<typename T>
double normalize2D(T **ar, int size1, int size2) {
    T sum = sum2D(ar, size1, size2);
    for (int i = 0; i < size1; ++i) {
        for (int j = 0; j < size2; ++j) {
            ar[i][j] /= sum;
        }
    }
    return sum;

}

template<typename T>
void bounded1D(T *source, T *low, T *upper, int size) {
    for (int k = 0; k < size; ++k) {
        if (source[k] > upper[k]) {
            source[k] = upper[k];
        }
        if (source[k] < low[k]) {
            source[k] = low[k];
        }
    }
}

template<typename T>
void bounded2D(T **source, T **low, T **upper, int size1, int size2) {
    for (int i = 0; i < size1; ++i) {
        for (int k = 0; k < size2; ++k) {
            if (source[i][k] > upper[i][k]) {
                source[i][k] = upper[i][k];
            }
            if (source[i][k] < low[i][k]) {
                source[i][k] = low[i][k];
            }
        }
    }

}

template<class C>
class MatrixView {
public:
    C *data;
    int rows;
    int cols;

    MatrixView(int rows, int cols, C *ptr)  {
        this->rows=rows;
        this->cols = cols;
        this->data=ptr;
    };

    C *operator[](int k) { return &(this->data[k * this->cols]); }
};

template<class T>
int getArrayLen(T &array) {
    return (sizeof(array) / sizeof(array[0]));
}


//void printAlpha(double **alpha, double *cn, int n_x, int n_stat) {
//    double c = 1;
//    for (int t = 0; t < n_x; ++t) {
//        c *= cn[t];
//        std::cout << t;
//        for (int i = 0; i < n_stat; ++i) {
//            std::cout << " " << alpha[t][i] * c;
//        }
//        std::cout << std::endl;
//    }
//}
//
//void printBeta(double **beta, double *cn, int n_x, int n_stat) {
//    double c = 1;
//    for (int t = n_x - 1; t >= 0; --t) {
//        c *= cn[t];
//        std::cout << t;
//        for (int i = 0; i < n_stat; ++i) {
//            std::cout << " " << beta[t][i] * c;
//        }
//        std::cout << std::endl;
//    }
//}

#endif //HMM_UTILS_H
