#Define the shell
SHELL:=/bin/bash

# Define the MPI compiler
MPICC = mpicc

# Define the source and output files
SRC = main.c
OUT = ./bin/ownBarrier

# List the modules you need to load
MODULES = module load devtoolset/10 mpi/open-mpi-4.1.6

# Compilation flags
#CFLAGS = -O3 -g

all: build

build:
    # Load the required modules
	$(MODULES)

    # Compile the MPI program
	$(MPICC) $(CFLAGS) -o $(OUT) $(SRC)

clean:
    # Remove the executable
	rm -f $(OUT)
