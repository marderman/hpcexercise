
MPICC=mpicc

.PHONY: build
build: ./bin/measureBandwith

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./bin/measureBandwith: ./main.c
	$(MPICC) -Wall -lm -o $@ $^ $(INC) $(LIB)
