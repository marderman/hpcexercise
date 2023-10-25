CUDA_ROOT=$(CUDA_HOME)
INC=-I./inc -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64
NVCC=nvcc
NVCC_FLAGS=-O2 --compiler-options "-O2 -Wall -Wextra"

.PHONY: build
build: ./bin/nullKernelAsync

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./bin/nullKernelAsync: ./src/nullKernelAsync.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)
