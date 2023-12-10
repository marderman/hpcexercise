#CUDA_ROOT=/opt/cuda-7.0
INC=-I./inc -I $(CUDA_PATH)/include
#LIB=-L$(CUDA_ROOT)/lib64
#NVCC=$(CUDA_ROOT)/bin/nvcc

.PHONY: build
build: ./bin/reduction

.PHONY: clean
clean:
	rm -f ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/reduction: ./src/main.cpp ./src/kernel.cu 
	$(NVCC) -O2 -arch=compute_60 -code=sm_70 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)
