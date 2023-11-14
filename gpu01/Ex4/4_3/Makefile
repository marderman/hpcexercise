
NVCC=nvcc
INC=-I./inc -I$(CUDA_PATH)/include
LIB=-L$(CUDA_PATH)/lib64

.PHONY: build
build: ./bin/memCpy

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./bin/memCpy: ./src/main.cpp ./src/kernel.cu
	$(NVCC) -arch=compute_60 -code=sm_70 -O2 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)
