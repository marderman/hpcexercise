NVCC=nvcc
INC=-I./inc #-I/usr/local/cuda/include
#LIB=-L/usr/local/cuda/lib64

.PHONY: build
build: ./bin/memCpy

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/memCpy: ./src/template.cpp ./src/kernel.cu
	$(NVCC) -O2 -arch=sm_75 --compiler-options "-O2 -Wall -Wextra" -o $@ $^ $(INC) $(LIB)