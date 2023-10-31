
MPICC=mpicc

.PHONY: build
build: ./bin/ownBarrier

.PHONY: clean
clean:
	rm ./bin/*

.PHONY: rebuild
rebuild: clean build

# List of modules to load
MODULES := devtoolset/10 mpi/open-mpi-4.1.6

load:
	@echo "Loading modules..."
	@for module in $(MODULES); do \
		module load $$module; \
	done
	@echo "Modules loaded successfully"

.PHONY: load

./bin/ownBarrier: ./main.c
	$(MPICC) -lm -o $@ $^ $(INC) $(LIB)