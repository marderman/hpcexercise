
MPICC=mpicc

.PHONY: build
build: ./bin/heatMap

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./bin/heatMap: ./src/main.cpp
	$(MPICC) -g -lm -O3 -o $@ $^ $(INC) $(LIB)