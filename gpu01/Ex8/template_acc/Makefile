CC=pgc++
ACCFLAGS = -acc -ta=tesla,time -Minfo=accel
INC=-I./inc
LIB=

.PHONY: build
build: ./bin/stencil 

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/stencil: ./src/main.cpp
	$(CC) -O2  $(ACCFLAGS) -o $@ $^ $(INC) $(LIB)
