#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <stdbool.h>

#define N 2048

//Function to transpose a matrix
void transpose(double** matrix, int size){
    for(int i; i<size; i++){
        for(int j; j<size; j++){
            double temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
        }
    }
}

//Function to multiply matrices
void multiply_matrix(double** matrix_a, double** matrix_b, double** result, bool transposed){
    if(transposed == true){
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int h = 0; h < N; h++) {
                result[i][j] += matrix_a[i][h] * matrix_b[j][h];
            }
        }
    }        
    }
    else{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int h = 0; h < N; h++) {
                result[i][j] += matrix_a[i][h] * matrix_b[h][j];
            }
        }
    }
    }
}

int main() {
    // Initialize matrices and timestamps
    time_t start, stop;
    double** matrix_a;
    double** matrix_b;
    double** result;

    // Generate random seed
    srand(time(NULL));

    //Dynamic memory allocation for matrices, this needs to be done otherwise the process exits with a segmentation fault :(
    matrix_a = (double**)malloc(N * sizeof(double*));
    matrix_b = (double**)malloc(N * sizeof(double*));
    result = (double**)malloc(N * sizeof(double*));

    for (int i = 0; i < N; i++) {
        matrix_a[i] = (double*)malloc(N * sizeof(double));
        matrix_b[i] = (double*)malloc(N * sizeof(double));
        result[i] = (double*)malloc(N * sizeof(double));
    }
    printf("Initialization is working\n");

    //Fill arrays with random data
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix_a[i][j] = ((double)rand()) / RAND_MAX;
            matrix_b[i][j] = ((double)rand()) / RAND_MAX;
        }
    }
    printf("Filling matrices is working\n");

    //Multiply the matrices
    start = time(NULL);
    multiply_matrix(matrix_a, matrix_b, result, false);
    stop = time(NULL);
    //printf("Calculating the result matrix is working\n");

    //Print out the time it took and GFLOPS/s
    double time_taken = (double) difftime(stop, start);
    double gflops = (2 * pow(N, 3)) / (time_taken*1e9); 
    printf("It took %f seconds to calculate the matrix multiply without transposing the matrix, the performance was %f GFLOPS/s\n", time_taken, gflops);

    //Transpose the second matrix
    transpose(matrix_b, N);

    // Multiply the matrices but this time one is transposed
    start = time(NULL);
    multiply_matrix(matrix_a, matrix_b, result, true);
    stop = time(NULL);
    //printf("Calculating the result matrix (transposed) is working\n");

    //Print out the time it took and GFLOPS/s
    time_taken = (double) difftime(stop, start);
    gflops = (2 * pow(N, 3)) / (time_taken*1e9); 
    printf("It took %f seconds to calculate the matrix multiply with transposing the second matrix, the performance was %f GFLOPS/s\n", time_taken, gflops);

    //Free dynamically allocated memory
    for (int i = 0; i < N; i++) {
        free(matrix_a[i]);
        free(matrix_b[i]);
        free(result[i]);
    }
    free(matrix_a);
    free(matrix_b);
    free(result);

    return 0;
}
