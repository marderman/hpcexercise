// 1gbyte = 1073741824 bytes, 1 int = 4 byte, 1gb = 268435456 int
// 1mbyte = 1048576 bytes, 1 int = 4 byte, 1mb = 262144 int
// 1kbyte = 1024 bytes, 1 int = 4 byte, 1kb = 256 int
#include <stdio.h>
#include "chTimer.h"

chTimerTimestamp start,stop;
double seconds;

int
main()
{
    for ( int i = 0; i < 10; i++ ) {
        int *hmem = (int*)malloc((i+1)* 256 * sizeof(int));
        int *dmem = (int*)cudaMalloc((i+1)* 256 * sizeof(int));
        chTimerGetTime( &start );
        cudaMemcpy(hmem, dmem, (i+1)* 256 * sizeof(int), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        chTimerGetTime( &stop );
        cudaFree(dmem);
        free(hmem);
        seconds = 1e12*chTimerElapsedTime( &start, &stop );
        printf( "%.2f us\n", (seconds/1e6));
        printf( "%.2f us\n", ((((i+1) * 1024)/1073741824)/seconds));
    }

    return 0;
}