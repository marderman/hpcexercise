#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>

#define N 128

void matrix_multiply(double *a, double *b, double *c, int rows_per_process) {
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double *a, *b, *c;
    double *local_a, *local_c;
    int *sendcounts, *offset;
    int rows_per_process, remainder_rows;
    double start, stop;
    //printf("Process %d is starting up....\n", rank);

    a = (double *)malloc(N * N * sizeof(double));   //I dont understand why this cannot be inside if rank == 0???!!!
    b = (double *)malloc(N * N * sizeof(double));
    c = (double *)malloc(N * N * sizeof(double));

    // Allocate memory for matrices
    if (rank == 0) {

        //Generate random seed
        srand(time(NULL));
 
        //Fill arrays with random data
        for (int i = 0; i < N*N; i++) {
                a[i] = ((double)rand()) / RAND_MAX;
                b[i] = ((double)rand()) / RAND_MAX;
        }
        printf("Random fill is working\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    rows_per_process = N / size;
    remainder_rows = N % size;

    sendcounts = (int *)malloc(size * sizeof(double));
    offset = (int *)malloc(size * sizeof(double));

    // Calculate sendcounts and offset for uneven distribution of rows
    for (int i = 0; i < size; i++) {
        if (i < remainder_rows) {
            sendcounts[i] = (rows_per_process + 1) * N;
            offset[i] = i * (rows_per_process + 1) * N;
        } else {
            sendcounts[i] = rows_per_process * N;
            offset[i] = (i - remainder_rows) * rows_per_process * N + remainder_rows * (rows_per_process + 1) * N;
        }
    }


    // Scatter matrix 'a' to all processes
    local_a = (double *)malloc(sendcounts[rank] * sizeof(double));
    MPI_Scatterv(a, sendcounts, offset, MPI_DOUBLE, local_a, sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate memory for local result matrix 'c'
    local_c = (double *)malloc(rows_per_process * N * sizeof(double));

    // Scatter matrix 'b' to all processes
    MPI_Bcast(b, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform local matrix multiplication
    if(rank==0){
        start = MPI_Wtime();
    }
    matrix_multiply(local_a, b, local_c, rows_per_process);

    MPI_Barrier(MPI_COMM_WORLD);
    
    if(rank==0){
        stop = MPI_Wtime();
    }

    // Gather results back to the master process
    MPI_Gatherv(local_c, rows_per_process * N, MPI_DOUBLE, c, sendcounts, offset, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Clean up
    free(local_a);
    free(local_c);
    free(sendcounts);
    free(offset);
    free(a);    //This also cannot be in if rank == 0
    free(b);
    free(c);

    if (rank == 0) {
        double time_taken = stop-start;
        double gflops = (2 *pow(N, 3)) / (time_taken*1e9); 
        printf("It took %lf seconds to calculate the matrix multiply without transposing the %d-size matrix, with %d threads, the performance was %lf GFLOPS/s\n", time_taken, N, size, gflops);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}
