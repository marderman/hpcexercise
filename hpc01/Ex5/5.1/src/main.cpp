#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <getopt.h>
#include <ncurses.h>

#define rows 8
#define columns 8
#define N_ITERATIONS 1000


void performComputation(float* currentGrid, float* outputGrid, int localSize);
void output(MPI_Comm comm, float* partialGrid, int localSize);
void initialDistribution();

MPI_Comm row_comm;
int n_processes, rank;


int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);

    //Get the global rank and global size

    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //Calculate the amount of rows each process has to compute (Set to minimum of 1)
    int rows_per_process = std::max(rows / n_processes, 1);

// Dimensions in Coordinate System
    int dims[1] = {n_processes};
    int periods[1] = {0};

    //Communicator for the rows
    MPI_Cart_create(MPI_COMM_WORLD,1,dims,periods,0, &row_comm);

    //Get the coordinates of the process in the Cartesian communicator
    //int coords[1];
    //MPI_Cart_coords(row_comm, rank, 1, coords);

    // Get neighbor information
    int leftNeighbor, rightNeighbor;
    MPI_Cart_shift(row_comm, 0, 1, &leftNeighbor, &rightNeighbor);

    printf("Hello from process %d, my neighbors are %d and %d\n",rank,leftNeighbor,rightNeighbor);

    //Each Process allocates the memory he is working on
    float* partialGrid = (float*)malloc((2+rows_per_process)*(columns+2)*sizeof(float));
    float* previousPartialGrid = (float*)malloc((2+rows_per_process)*sizeof(float));
    printf("Allocated Memory\n");
    for (size_t i = 0; i < rows_per_process*columns; i++)
    {
        partialGrid[i] = rank;
        /* code */
    }
    printf("Initialized Memory\n");
    fflush(stdout);
    MPI_Barrier(row_comm);

    //Distribute the Start Data
    if (rank == 0) {
    //MPI_Scatter();

    }
    //Calculate the dissipation
    for (size_t i = 0; i < N_ITERATIONS; i++)
    {
        //performComputation(currentGrid,previousGrid,rows_per_process);
    }

    output(row_comm, partialGrid, rows_per_process*columns);

    //Get back the results
    

    free(partialGrid);
    free(previousPartialGrid);
    MPI_Comm_free(&row_comm);

    MPI_Finalize();
}

void output(MPI_Comm comm, float* partialGrid, int localSize)
{
    float* recv_wholeGrid;
    if (rank == 0) {
        recv_wholeGrid = (float*) malloc(columns*rows*sizeof(float));
    }
    MPI_Barrier(comm);
    //Get the rows of each process (partial Grids) and combine to one large grid
    MPI_Gather(partialGrid,columns, MPI_FLOAT,
                recv_wholeGrid,columns,MPI_FLOAT,
                0,row_comm);
    MPI_Barrier(comm);

    if(rank == 0 ) {
    for (size_t i = 0; i < columns*rows; i++)
    {
        printf("HAHAHA: %f\n", recv_wholeGrid[i]);
        /* code */
    }
    
    free(recv_wholeGrid);
    }
}

void performComputation(float* currentGrid, float* outputGrid, int localSize) {
 int cart_rank, cart_size;

 MPI_Comm_rank(row_comm,&cart_rank);
 MPI_Comm_size(row_comm,&cart_size);

 if (cart_rank == 0) {

 }
 else if (cart_rank < cart_size) {

 }
 else {

 }
 

}

void initialDistribution() {

}

