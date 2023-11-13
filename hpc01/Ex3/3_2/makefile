
MPICC=mpicc

.PHONY: build
build: ./bin/matrixMultiplyS

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./bin/matrixMultiplyS: ./main.c
	$(MPICC) -lm -o $@ $^ $(INC) $(LIB)