
MPICC=gcc

.PHONY: build
build: ./bin/heatRelax

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./bin/heatRelax: ./main.c
	$(MPICC) -Wall -O3 -lm -lncurses -o $@ $^ $(INC) $(LIB)
