#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <getopt.h>
#include <ncurses.h>
#include <math.h>
#include <unistd.h>

//#define DEBUG

MPI_Comm row_comm;
const static float TIMESTEP = 1e-3;	  // s
const static float GAMMA = 6.673e-11; // (Nm^2)/(kg^2)
int NUM_BODIES, N_ITERATIONS, n_processes, rank, objects_per_process;

struct Body_t
{
	float4 *posMass;  /* x = x */
					  /* y = y */
					  /* z = z */
					  /* w = Mass */
	float3 *velocity; /* x = v_x */
					  /* y = v_y */
					  /* z = v_z */

	Body_t() : posMass(NULL), velocity(NULL) {}
};

int main(int argc, char *argv[])
{
    double start, stop;
    NUM_BODIES = 256;
    N_ITERATIONS = 100;

    int opt;
    while ((opt = getopt(argc, argv, "hw:l:n:")) != -1)
    {
        switch (opt)
        {
        case 'h':
            std::cout << "example -b <num_bodies> -n <num_iterations>" << std::endl;
            std::exit(EXIT_SUCCESS);
        case 'b':
            NUM_BODIES = std::atoi(optarg);
            break;
        case 'n':
            N_ITERATIONS = std::atoi(optarg);
            break;
        default:
            std::cerr << "Usage: example -b <num_bodies> -n <num_iterations>" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    MPI_Init(&argc, &argv);

    // Get the global rank and global size

    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Calculate the amount of rows each process has to compute (Set to minimum of 1)
    objects_per_process = std::max(NUM_BODIES / n_processes, 1);

    // Dimensions in Coordinate System
    int dims[1] = {n_processes};
    int periods[1] = {1}; // Forms circle in the cart if enabled

    // Communicator for the rows
    MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, 0, &row_comm);

    // Get the coordinates of the process in the Cartesian communicator
    // int coords[1];
    // MPI_Cart_coords(row_comm, rank, 1, coords);

    // Get neighbor information
    int leftNeighbor, rightNeighbor;
    MPI_Cart_shift(MPI_COMM_WORLD, 0, 1, &leftNeighbor, &rightNeighbor);
#ifdef DEBUG
    printf("Hello from process %d, my neighbors are %d and %d\n", rank, leftNeighbor, rightNeighbor);
#endif

    // Each Process allocates the memory it is working on
    float *partialGrid = (float *)malloc((2 + rows_per_process) * (columns) * sizeof(float));         // +2 for ghost rows
    float *previousPartialGrid = (float *)malloc((2 + rows_per_process) * (columns) * sizeof(float)); // For calculating matrices without extra copy

    initialDistribution(partialGrid, previousPartialGrid); // Initialize arrays
    MPI_Barrier(row_comm);
#ifdef DEBUG
    fflush(stdout);
    output(row_comm, partialGrid, rows_per_process * columns);
#endif

    if (rank == 0)
    {
        start = MPI_Wtime();
    }

    // Calculate the dissipation
    for (size_t i = 0; i < N_ITERATIONS; i++)
    {
        if (i % 2 == 0)
        {
            performComputation(partialGrid, previousPartialGrid, rows_per_process, leftNeighbor, rightNeighbor, i);
        }
        if (i % 2 == 1)
        {
            performComputation(previousPartialGrid, partialGrid, rows_per_process, leftNeighbor, rightNeighbor, i);
        }
    }
    MPI_Barrier(row_comm);

    if (rank == 0)
    {
        stop = MPI_Wtime();
        double time_taken = (stop - start) / N_ITERATIONS;
        printf("The time for a n-body problem with %d bodies took %lf seconds\n", NUM_BODIES, time_taken);
    }

// Get back the results
#ifdef DEBUG
    output(row_comm, partialGrid, rows_per_process * columns);
#endif

    free(partialGrid);
    free(previousPartialGrid);
    MPI_Comm_free(&row_comm);

    MPI_Finalize();
}

void initializeBodies(){

}

void computeBodies(){

}