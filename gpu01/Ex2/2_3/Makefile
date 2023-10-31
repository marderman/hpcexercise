CUDA_ROOT=/opt/cuda-12.0
INC=-I./inc -I$(CUDA_ROOT)/include
LIB=-L$(CUDA_ROOT)/lib64
NVCC=nvcc
NVCC_FLAGS=-O2 --compiler-options "-O2 -Wall -Wextra"

.PHONY: build
build: ./ex2_3.exe

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./ex2_3.exe: ./ex2_3.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(INC) $(LIB)
