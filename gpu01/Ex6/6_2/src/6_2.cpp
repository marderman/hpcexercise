#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chTimer.hpp>


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

    // Populate array with ones
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        array[i] = 1;
    }

    ChTimer kernelTimer;

    kernelTimer.start();

    // Perform global sum reduction with modification for computation
    double result = global_sum_reduction(array, size);

    kernelTimer.stop();

    printf("Global sum: %f\n", result);
    printf("Run time: %f milliseconds\n", 1e3 * kernelTimer.getTime());

    if (kernelTimer.getTime() < 1e-9) {
        printf("Bandwidth: Infinite elements per second (very short run time)\n");
    } else {
        printf("Bandwidth: %f elements per second\n", size / kernelTimer.getTime());
    }

    free(array);

    return 0;
}
