#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void matrix_multiply(int size, int A[size][size], int B[size][size], int C[size][size]) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i][j] = 0;
            for (int k = 0; k < size; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main() {
    // L3 cache size in kilobytes
    size_t l3_cache_size_kb = 10240; // 10 MB

    // Calculate the matrix size for which the data will exceed the L3 cache size
    int size = 1;
    while (size * size * sizeof(int) <= l3_cache_size_kb * 1024) {
        size += 100; // Increment by a suitable value based on your system and problem size
    }

    // Allocate memory for matrices
    int (*A)[size] = malloc(sizeof(int[size][size]));
    int (*B)[size] = malloc(sizeof(int[size][size]));
    int (*C)[size] = malloc(sizeof(int[size][size]));

    // Initialize matrices
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            A[i][j] = i + 1;
            B[i][j] = j + 1;
        }
    }

    // Measure execution time
    clock_t start_time = clock();
    matrix_multiply(size, A, B, C);
    clock_t end_time = clock();
    double elapsed_time = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;

    // Calculate sustained GFLOP/s
    double gflops = (2.0 * size * size * size) / (elapsed_time * 1e9);

    // Print results
    printf("Matrix Size: %d x %d\n", size, size);
    printf("Execution Time: %f seconds\n", elapsed_time);
    printf("Sustained GFLOP/s: %f\n", gflops);

    // Free allocated memory
    free(A);
    free(B);
    free(C);

    return 0;
}
