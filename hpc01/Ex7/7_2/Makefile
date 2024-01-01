
MPICC=gcc

.PHONY: build
build: ./bin/nBody

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./bin/nBody: ./main.c
	$(MPICC) -Wall -O3 -lm -lncurses -o $@ $^ $(INC) $(LIB)
