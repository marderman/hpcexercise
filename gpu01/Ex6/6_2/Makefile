#CUDA_ROOT=/opt/cuda-7.0
INC=-I./inc #-I. -I$(CUDA_ROOT)/include
#LIB=-L$(CUDA_ROOT)/lib64
#NVCC=$(CUDA_ROOT)/bin/nvcc

.PHONY: build
build: ./bin/reduction

.PHONY: clean
clean:
	rm -f ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/reduction: ./src/6_2.cpp
	$(NVCC) -O2 -arch=sm_75 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)
