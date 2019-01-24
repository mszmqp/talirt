cmake_minimum_required(VERSION 3.12)
project(hmm)

set(CMAKE_CXX_STANDARD 11)
INCLUDE_DIRECTORIES(/usr/local/Cellar/gsl/2.5/include/)
LINK_DIRECTORIES(/usr/local/Cellar/gsl/2.5/lib/)
#add_library(hmm _hmm.cpp man.cpp)
add_executable(bkt _hmm.cpp man.cpp _bkt.cpp _bkt.h utils.h TrainHelper.cpp TrainHelper.h)
TARGET_LINK_LIBRARIES(bkt gsl gslcblas m)