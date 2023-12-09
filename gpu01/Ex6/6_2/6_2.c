#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double global_sum_reduction(int* array, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        // Perform some actual computation on the array elements
        sum += array[i] * array[i];
    }
    return sum;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);

    int *array = (int *)malloc(size * sizeof(int));
    if (array == NULL) {
        printf("Memory allocation error.\n");
        return 1;
    }

    // Populate array with random values
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        array[i] = rand();
    }

    clock_t start_time = clock();

    // Perform global sum reduction with modification for computation
    double result = global_sum_reduction(array, size);

    clock_t end_time = clock();
    double cpu_time_used = ((double) (end_time - start_time)) / CLOCKS_PER_SEC;

    printf("Global sum: %f\n", result);
    printf("Run time: %f seconds\n", cpu_time_used);

    if (cpu_time_used < 1e-9) {
        printf("Bandwidth: Infinite elements per second (very short run time)\n");
    } else {
        printf("Bandwidth: %f elements per second\n", size / cpu_time_used);
    }

    free(array);

    return 0;
}
