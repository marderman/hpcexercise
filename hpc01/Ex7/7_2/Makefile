
MPICC=mpic++

.PHONY: build
build: ./bin/nBody

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./bin/nBody: ./src/main.cpp
	$(MPICC) -Wall -O3 -lm -o $@ $^ $(INC) $(LIB)
