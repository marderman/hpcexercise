#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>


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
/*double matrix_multiply(double array1[2048][2048], double array2[2048][2048]){
    int array_x_size = sizeof(array1) / sizeof(array1[0]);
    int array_y_size = sizeof(array1[0])/sizeof(array1[0][0]);
    if(array_x_size != sizeof(array2) / sizeof(array2[0])){
        printf("The arrays are not of the same size");
        return;
    }
    if(array_y_size != sizeof(array2[0])/sizeof(array2[0][0]);){
        printf("The arrays are not of the same size");
        return;
    }    
    double result[2048][2048];
    for(int i; i<4194304; i++){
        for(int j; j<4194304; j++){
            for(int h; h<4194304; h++){
                result[i][j] = result[i][j] + array1[i][h] * array2[h][j];
            }
        }
    }
    return result;
}*/

int main() {
    // Initialize matrices and timestamps
    time_t start, stop;
    double** matrix_a;
    double** matrix_b;
    double** result;

    // Generate random seed
    srand(time(NULL));

    // Dynamic memory allocation for matrices
    matrix_a = (double**)malloc(2048 * sizeof(double*));
    matrix_b = (double**)malloc(2048 * sizeof(double*));
    result = (double**)malloc(2048 * sizeof(double*));

    for (int i = 0; i < 2048; i++) {
        matrix_a[i] = (double*)malloc(2048 * sizeof(double));
        matrix_b[i] = (double*)malloc(2048 * sizeof(double));
        result[i] = (double*)malloc(2048 * sizeof(double));
    }
    printf("Initialization is working\n");

    // Fill arrays with random data
    for (int i = 0; i < 2048; i++) {
        for (int j = 0; j < 2048; j++) {
            matrix_a[i][j] = ((double)rand()) / RAND_MAX;
            matrix_b[i][j] = ((double)rand()) / RAND_MAX;
        }
    }
    printf("Filling matrices is working\n");

    // Multiply the matrices
    start = time(NULL);
    for (int i = 0; i < 2048; i++) {
        for (int j = 0; j < 2048; j++) {
            result[i][j] = 0.0;  // Initialize result element
            for (int h = 0; h < 2048; h++) {
                result[i][j] += matrix_a[i][h] * matrix_b[h][j];
            }
        }
    }
    stop = time(NULL);
    printf("Calculating the result matrix is working\n");

    // Print out the time it took and GFLOPS/s
    double time_taken = difftime(stop, start);
    double gflops = (2 * pow(2048, 3)) / (time_taken * 1e9); 
    printf("It took %f seconds to calculate the matrix multiply without transposing the matrix, the performance was %f GFLOPS/s\n", time_taken, gflops);

    //Transpose the second matrix
    transpose(matrix_b, 2048);

    // Multiply the matrices but this time one is transposed
    start = time(NULL);
    for (int i = 0; i < 2048; i++) {
        for (int j = 0; j < 2048; j++) {
            result[i][j] = 0.0;  // Initialize result element
            for (int h = 0; h < 2048; h++) {
                result[i][j] += matrix_a[i][h] * matrix_b[j][h];
            }
        }
    }
    stop = time(NULL);
    printf("Calculating the result matrix (transposed) is working\n");

    // Print out the time it took and GFLOPS/s
    time_taken = difftime(stop, start);
    gflops = (2 * pow(2048, 2)) / (time_taken * 1e9); 
    printf("It took %f seconds to calculate the matrix multiply with transposing the second matrix, the performance was %f GFLOPS/s\n", time_taken, gflops);

    // Free dynamically allocated memory
    for (int i = 0; i < 2048; i++) {
        free(matrix_a[i]);
        free(matrix_b[i]);
        free(result[i]);
    }
    free(matrix_a);
    free(matrix_b);
    free(result);

    return 0;
}
