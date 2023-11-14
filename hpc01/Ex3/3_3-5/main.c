#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>
#include <mpi.h>

#define N 2048

//Function to multiply matrices
void multiply_matrix(double matrix_a[N][N], double matrix_b[N][N], double result[N][N], int start_row, int end_row, bool transposed){
    //Multiplication for a transposed matrix
    if(transposed == true){
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            for (int h = 0; h < N; h++) {
                result[i][j] += matrix_a[i][h] * matrix_b[j][h];
            }
        }
    }        
    }
    else{
    //Multiplication for regualar matrices
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            for (int h = 0; h < N; h++) {
                result[i][j] += matrix_a[i][h] * matrix_b[h][j];
            }
        }
    }
    }
}

int main(int argc, char *argv[]) {
    
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Divide the rows among the threads
    int rows_per_process = N / size;
    int start_row = rank * rows_per_process;
    int end_row = (rank + 1) * rows_per_process;

    //Initialize the arrays
    double matrix_a[N][N];
    double matrix_b[N][N];
    double result[N][N];
    time_t start, stop;

    if(rank == 0){
        // Generate random seed
        srand(time(NULL));
    
        // Fill arrays with random data
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix_a[i][j] = ((double)rand()) / RAND_MAX;
                matrix_b[i][j] = ((double)rand()) / RAND_MAX;
            }
        }
    }

    //Distribute the matrices among the threads
    MPI_Bcast(matrix_a, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrix_b, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //Set a start point capturing the start of the calculation in time
    if (rank == 0) {
       start = time(NULL); 
    }

    //Each process performs a part of the matrix multiplication
    multiply_matrix(matrix_a, matrix_b, result, start_row, end_row, false);

    //Make sure to synchronize
    MPI_Barrier(MPI_COMM_WORLD);

    //Gather the results back to the root process (process 0)
    MPI_Gather(result + start_row, rows_per_process * N, MPI_DOUBLE, result, rows_per_process * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    //Set a stop point capturing the end of the calculation in time and output the time and GFLOPS
    if (rank == 0) {
        stop = time(NULL);
        double time_taken = (double) difftime(stop, start);
        double gflops = (2 * pow(N, 3)) / (time_taken*1e9); 
        printf("It took %f seconds to calculate the matrix multiply without transposing the matrix, the performance was %f GFLOPS/s\n", time_taken, gflops);
    }

    return 0;
}
