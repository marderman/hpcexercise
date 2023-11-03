#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "chTimer.h"


int main() {
    const int minSize = 1024;  // 1 KB
    const int maxSize = 1024 * 1024 * 1024;  // 1 GB
    const int numIterations = 500;
    chTimerTimestamp start,stop;

    for (long size = minSize; size <= maxSize; size *= 2) {
        if(size >= maxSize){ size = maxSize; }
        float* h_data = (float*)malloc((size/4) * sizeof(float));
        float* d_data;
        cudaMalloc((void**)&d_data, (size/4) * sizeof(float));

        // Measure host-to-device bandwidth
        chTimerGetTime( &start );
        for (int iter = 0; iter < numIterations; iter++) {
            cudaMemcpy(d_data, h_data, (size/4) * sizeof(float), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
        }
        chTimerGetTime( &stop );
        double elapsedSeconds = chTimerElapsedTime( &start, &stop );
        double hostToDeviceBandwidth = ((size/4) * sizeof(float) * numIterations) / (elapsedSeconds * 1e9); // GB/s

        // Measure device-to-host bandwidth
        chTimerGetTime( &start );
        for (int iter = 0; iter < numIterations; iter++) {
            cudaMemcpy(h_data, d_data, (size/4) * sizeof(float), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
        }
        chTimerGetTime( &stop );
        elapsedSeconds = chTimerElapsedTime( &start, &stop );
        double deviceToHostBandwidth = ((size/4) * sizeof(float) * numIterations) / (elapsedSeconds * 1e9); // GB/s

        // Measure device-to-device bandwith
        float* d_data_2;
        cudaMalloc((void**)&d_data_2, (size/4) * sizeof(float));
        chTimerGetTime( &start );
        for (int iter = 0; iter < numIterations; iter++) {
            cudaMemcpy(d_data_2, d_data, (size/4) * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaDeviceSynchronize();
        }
        chTimerGetTime( &stop );
        elapsedSeconds = chTimerElapsedTime( &start, &stop );
        double deviceToDeviceBandwidth = ((size/4) * sizeof(float) * numIterations) / (elapsedSeconds * 1e9); // GB/s

        printf("Data Size: %d KB\tHost-to-Device Bandwidth: %lf GB/s\tDevice-to-Host Bandwidth: %lf GB/s\tDevice-to-Device Bandwidth: %lf GB/s\n", size / 1024, hostToDeviceBandwidth, deviceToHostBandwidth, deviceToDeviceBandwidth);

        // Clean up
        cudaFree(d_data);
        cudaFree(d_data_2);
        free(h_data);
        cudaDeviceSynchronize();
    }

    return 0;
}