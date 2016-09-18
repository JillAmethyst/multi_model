# Modify DLIB_PATH and EIGEN_PATH to adapt local environment

# Dlib dependency
DLIB_PATH := /share/home/guan/dlib-18.16
INC_PATH := -I$(DLIB_PATH)
LIB := $(DLIB_PATH)/examples/build/dlib_build/libdlib.a

# Eigen dependency
EIGEN_PATH := /share/home/guan/eigen-3.2.8
INC_PATH += -I$(EIGEN_PATH)

# rapidjson dependency
INC_PATH += -I./include


# rapidjson dependency
INC_PATH += -I/share/home/guan/bin/include

# Basic parameters
#CXX = g++
#CXX = icc   -mkl  -m64 -w   -Wl,--start-group "/share/home/guan/intel/mkl/lib/intel64"/libmkl_intel_lp64.a "/share/home/guan/intel/mkl/lib/intel64"/libmkl_sequential.a "/share/home/guan/intel/mkl/lib/intel64"/libmkl_core.a -Wl,--end-group  -lpthread -ldl -lm 
#CXX = icc -mkl
#CXX = mpicxx  -I/share/home/guan/intel/compilers_and_libraries_2016.3.210/linux/mkl/include/   -m64 -w   -Wl,--start-group "/share/home/guan/intel/mkl/lib/intel64"/libmkl_intel_lp64.a "/share/home/guan/intel/mkl/lib/intel64"/libmkl_sequential.a "/share/home/guan/intel/mkl/lib/intel64"/libmkl_core.a -Wl,--end-group  -lpthread -ldl -lm  /share/home/guan/intel/compilers_and_libraries_2016.3.210/linux/mkl/lib/intel64/libmkl_blacs_intelmpi_lp64.a
CXX = mpicxx
CXX_FLAGS =  -DUSE_AVX_INSTRUCTIONS -DUSE_SSE2_INSTRUCTIONS -DUSE_SSE4_INSTRUCTIONS #-DDLIB_USE_LAPACK -DDLIB_USE_BLAS #-DEIGEN_USE_MKL_ALL  -I/share/home/guan/intel/compilers_and_libraries_2016.3.210/linux/mkl/include/ 
LINK_FLAGS = -L/share/home/guan/bin/lib -lglog /share/home/guan/OpenBlasMPI/lib/libopenblas_haswellp-r0.2.18.a /share/home/guan/intel/compilers_and_libraries_2016.3.210/linux/mkl/lib/intel64/libmkl_blacs_intelmpi_lp64.a


# OpenMP switch
USE_OpenMP := 1
ifeq ($(USE_OpenMP), 1)
    CXX_FLAGS += -DUSE_OMP -fopenmp  
    LINK_FLAGS += -fopenmp
endif

# debug switch
DEBUG := 0
ifeq ($(DEBUG), 1)
    CXX_FLAGS += -g -O0
else
    CXX_FLAGS += -O2
endif

SRC = ADMM_Dlib.cpp ClassificationMultiClassDecFuxJoint.cpp data_function.cpp matlab_function.cpp OnlineUnsupTaskDrivDicLeaJointC.cpp finalADMMfeature.cpp computeWandB.cpp ComputeCost.cpp SGDMultiClassQuadC.cpp multimodal_config_parser.cpp test_main.cpp
OBJ = $(SRC:.cpp=.o)

test: $(OBJ)
	$(CXX) -o $@ $^ $(LIB) $(LINK_FLAGS)

%.o: %.cpp
	$(CXX) -c $< $(INC_PATH) $(CXX_FLAGS)

tmp: test_tmp.cpp
	$(CXX) -o $@ $^ $(INC_PATH) $(CXX_FLAGS) $(LIB) $(LINK_FLAGS)

clean:
	rm -rf *.o test tmp
.PHONY: clean
