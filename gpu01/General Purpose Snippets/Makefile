
NVCC=nvcc

.PHONY: build
build: ./bin/globalMemory

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

# List of modules to load
MODULES := devtoolset/10 cuda/12.0

load:
	@#!/bin/bash
	@echo "Loading modules..."
	@for module in $(MODULES); do \
		module load $$module; \
	done
	@echo "Modules loaded successfully"

.PHONY: load

./bin/globalMemory: ./main.cu
	$(NVCC) -o $@ $^ $(INC) $(LIB)
