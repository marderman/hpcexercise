CC=pgc++
ACCFLAGS = -acc -ta=tesla,time -Minfo=accel -Minline
INC=-I./inc
LIB=

.PHONY: build
build: ./bin/gameoflife 

.PHONY: clean
clean:
	rm ./bin/*
	
.PHONY: rebuild
rebuild: clean build

./bin/gameoflife: ./src/gameoflife.c
	$(CC) -O2  $(ACCFLAGS) -o $@ $^ $(INC) $(LIB)
