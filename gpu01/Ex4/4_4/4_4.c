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
    int sizes[] = {5, 10, 50, 100, 500,1000,2000};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    FILE *csv_file = fopen("runtimes.csv", "w");
    if (csv_file == NULL) {
        perror("Error opening file");
        return 1;
    }

    fprintf(csv_file, "Problem Size,Execution Time (s)\n");

    for (int i = 0; i < num_sizes; i++) {
        int size = sizes[i];

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

        // Print results to CSV
        fprintf(csv_file, "%d,%f\n", size, elapsed_time);

        // Free allocated memory
        free(A);
        free(B);
        free(C);
    }

    fclose(csv_file);

    return 0;
}
