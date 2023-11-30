#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <getopt.h>
#include <ncurses.h>
#include <math.h>

#define rows 12
#define columns 12
#define N_ITERATIONS 1000

void performComputation(float *currentGrid, float *outputGrid, int localSize, int leftNeighbor, int rightNeighbor, int numIteration);
void output(MPI_Comm comm, float *partialGrid, int localSize);
void initialDistribution(float *previousGrid, float *previousPartialGrid);

MPI_Comm row_comm;
int n_processes, rank, rows_per_process;

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);

    // Get the global rank and global size

    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Dont even start if there are too many processes
    if (n_processes > rows)
    {
        if (rank == 0)
        {
            printf("Too many processes or too little gridsize. How dare you!\n");
        }
        MPI_Finalize();
        return 1;
    }

    // Calculate the amount of rows each process has to compute (Set to minimum of 1)
    rows_per_process = std::max(rows / n_processes, 1);

    // Dimensions in Coordinate System
    int dims[1] = {n_processes};
    int periods[1] = {0}; // Forms circle in the cart if enabled

    // Communicator for the rows
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &row_comm);

    // Get the coordinates of the process in the Cartesian communicator
    // int coords[1];
    // MPI_Cart_coords(row_comm, rank, 1, coords);

    // Get neighbor information
    int leftNeighbor, rightNeighbor;
    MPI_Cart_shift(row_comm, 0, 1, &leftNeighbor, &rightNeighbor);
    printf("Hello from process %d, my neighbors are %d and %d\n", rank, leftNeighbor, rightNeighbor);

    // Each Process allocates the memory it is working on
    float *partialGrid = (float *)malloc((2 + rows_per_process) * (columns) * sizeof(float));         // +2 for ghost rows
    float *previousPartialGrid = (float *)malloc((2 + rows_per_process) * (columns) * sizeof(float)); // For calculating matrices without extra copy
    printf("Allocated Memory\n");

    initialDistribution(partialGrid, previousPartialGrid); // Initialize arrays
    MPI_Barrier(row_comm);

    printf("Initialized Memory\n");
    fflush(stdout);
    MPI_Barrier(row_comm);

    // Calculate the dissipation
    for (size_t i = 0; i < N_ITERATIONS; i++)
    {
        if(i % 2 == 0){performComputation(partialGrid, previousPartialGrid, rows_per_process, leftNeighbor, rightNeighbor, i);}
        if(i % 2 == 1){performComputation(previousPartialGrid, partialGrid, rows_per_process, leftNeighbor, rightNeighbor, i);}
    }
    MPI_Barrier(row_comm);
    output(row_comm, partialGrid, rows_per_process * columns);

    // Get back the results

    free(partialGrid);
    free(previousPartialGrid);
    MPI_Comm_free(&row_comm);

    MPI_Finalize();
}

void output(MPI_Comm comm, float *partialGrid, int localSize)
{
    float *recv_wholeGrid;
    if (rank == 0)
    {
        recv_wholeGrid = (float *)malloc(columns * rows * sizeof(float));
    }
    MPI_Barrier(comm);
    // Get the rows of each process (partial Grids) and combine to one large grid
    if (rank < rows)
    {
        MPI_Gather(partialGrid, rows_per_process * columns, MPI_FLOAT,
                   recv_wholeGrid, rows_per_process * columns, MPI_FLOAT,
                   0, row_comm);
    }
    MPI_Barrier(comm);

    if (rank == 0)
    {
        for (size_t i = 0; i < columns * rows; i++)
        {
            printf("HAHAHA: %f\n", recv_wholeGrid[i]);
            /* code */
        }

        free(recv_wholeGrid);
    }
}

void performComputation(float *currentGrid, float *outputGrid, int localSize, int leftNeighbor, int rightNeighbor, int numIteration)
{
    // Do not perform send operation for first calculation
    if (numIteration != 0)
    {
        // Check for edgecase, pun intended
        if (rightNeighbor != -2 && leftNeighbor != -2)
        {
            MPI_Isend(currentGrid+(localSize * rows), columns, MPI_FLOAT, rightNeighbor, 0, row_comm, NULL);      // Send last row
            MPI_Isend(currentGrid+rows, columns, MPI_FLOAT, leftNeighbor, 0, row_comm, NULL);                   // Send first row
            MPI_Recv(currentGrid+((localSize + 1) * rows), columns, MPI_FLOAT, rightNeighbor, 0, row_comm, NULL); // Receive right ghost layer
            MPI_Recv(currentGrid, columns, MPI_FLOAT, leftNeighbor, 0, row_comm, NULL);                       // Receive left ghost layer
        }
        else if (rightNeighbor == -2)
        {
            MPI_Isend(currentGrid+rows, columns, MPI_FLOAT, leftNeighbor, 0, row_comm, NULL); // Send first row
            MPI_Recv(currentGrid, columns, MPI_FLOAT, leftNeighbor, 0, row_comm, NULL);     // Receive left ghost layer
        }
        else if (leftNeighbor == -2)
        {
            MPI_Isend(currentGrid+(localSize * rows), columns, MPI_FLOAT, rightNeighbor, 0, row_comm, NULL);      // Send last row
            MPI_Recv(currentGrid+((localSize + 1) * rows), columns, MPI_FLOAT, rightNeighbor, 0, row_comm, NULL); // Receive right ghost layer
        }
    }

    // For loop to go through the row of the partial grid and perform calculation
    for (size_t i = columns; i < (localSize * columns) - columns; i++)
    {
        // Check for edgecase
        if (i % columns == 0)   // No element to the left
        {
            outputGrid[i] = currentGrid[i] + 0.24 * ((-4.0) * currentGrid[i] + currentGrid[i + 1] + currentGrid[i - columns] + currentGrid[i + columns]);
        }
        else if (i % columns == columns - 1)    // No element to the right
        {
            outputGrid[i] = currentGrid[i] + 0.24 * ((-4.0) * currentGrid[i] + currentGrid[i - 1] + currentGrid[i - columns] + currentGrid[i + columns]);
        }
        else
        {
            outputGrid[i] = currentGrid[i] + 0.24 * ((-4.0) * currentGrid[i] + currentGrid[i + 1] + currentGrid[i - 1] + currentGrid[i - columns] + currentGrid[i + columns]);
        }
    }
}

void initialDistribution(float *partialGrid, float *previousPartialGrid)
{
    for (size_t i = 0; i < rows_per_process * columns; i++)
    {
        // partialGrid[i] = rank; // Check if process writes in correct addresspace
        if (rank == 0 && (double)i >= (columns * rows_per_process / 4.0) && (double)i <= (columns * rows_per_process * 3.0) / 4.0)
            partialGrid[i] = 127.0;
        else
            partialGrid[i] = 0.0;

        previousPartialGrid[i] = 0.0;
    }
}
