
MPICC=mpicc

.PHONY: build
build: ./bin/matrixMultiplyP

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./bin/matrixMultiplyP: ./main.c
	$(MPICC) -g -lm -o $@ $^ $(INC) $(LIB)