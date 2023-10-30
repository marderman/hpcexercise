
MPICC=mpicc


.PHONY: build
build: ./bin/measureLatency

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./bin/measureLatency: ./main.c
	$(MPICC) -lm -o $@ $^ $(INC) $(LIB)
