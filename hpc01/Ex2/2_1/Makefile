
MPICC=mpicc


.PHONY: build
build: ./bin/ringCommunication

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

./bin/ringCommunication: ./src/main.c
	$(MPICC) -o $@ $^ $(INC) $(LIB)
