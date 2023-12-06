#CUDA_ROOT=/opt/cuda-7.0
INC=-I./inc #-I. -I$(CUDA_ROOT)/include
#LIB=-L$(CUDA_ROOT)/lib64
#NVCC=$(CUDA_ROOT)/bin/nvcc
NVCC_FLAGS=-O2 -arch=sm_75 --compiler-options "-O2 -Wall -Wextra"

.PHONY: build
build: ./bin/reduction

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/reduction: ./src/main.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ ./src/main.cu $(INC) $(LIB)