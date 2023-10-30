/*
 *
 * nullKernelAsync.cu
 *
 * Microbenchmark for throughput of asynchronous kernel launch.
 *
 * Build with: nvcc -I ../chLib <options> nullKernelAsync.cu
 * Requires: No minimum SM requirement.
 *
 * Copyright (c) 2011-2012, Archaea Software, LLC.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions 
 * are met: 
 *
 * 1. Redistributions of source code must retain the above copyright 
 *    notice, this list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright 
 *    notice, this list of conditions and the following disclaimer in 
 *    the documentation and/or other materials provided with the 
 *    distribution. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <stdio.h>
#include "chTimer.h"

// __global__ void NullKernel(void)
// {

// }

__global__ void TimeKernel(long long int endtime, int n)
{
    endtime = 0;
    int i = threadIdx.x;
    if (i < n)
    {
        long long int start = clock64();

        endtime = clock64() - start;
        printf("endtime: %lld", endtime);
    }
}

__global__ void ActiveWaitKernel(long long int timethd, long long int* endtime, int n)
{
    int i = threadIdx.x;
    if (i < n)
    {
        long long int start = clock64();
        long long int end;
        do{
            end = clock64() - start;
        }while(end < timethd);
        *endtime = end;
    }
}

int main()
{
    long long int h_time = 0;
    long long int hh_time = 0;
    long long int* d_time;
    long long int* dd_time;
    long long int h_clockcycleswait = 0;
    long long int* d_clockcycleswait;
    int N = 1;

    cudaMalloc(&d_time, sizeof(long long int));
    cudaMalloc(&dd_time, sizeof(long long int));
    cudaMalloc(&d_clockcycleswait, sizeof(long long int));

    cudaMemcpy(&d_time, &h_time, sizeof(long long int), cudaMemcpyHostToDevice);
    cudaMemcpy(&dd_time, &hh_time, sizeof(long long int), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_clockcycleswait, &h_clockcycleswait, sizeof(long long int), cudaMemcpyHostToDevice);

    const int cIterations = 1;
    printf( "Measuring asynchronous launch time... " ); fflush( stdout );

    chTimerTimestamp start, stop;

    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) 
    {
        TimeKernel<<<1,N>>>(*dd_time, N);
    }
    cudaDeviceSynchronize();
    chTimerGetTime( &stop );
    

    double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
    double usPerLaunch = microseconds / (float) cIterations;
  
    printf( "%.5f us\n", usPerLaunch );
  
    double usPerLaunch2Kernel = 0;
    
    do
    {
        h_clockcycleswait += 2000;
        cudaMemcpy(&d_clockcycleswait, &h_clockcycleswait, sizeof(long long int), cudaMemcpyHostToDevice);

        chTimerGetTime( &start );
        for ( int i = 0; i < cIterations; i++ ) 
        {
            ActiveWaitKernel<<<1,N>>>(*d_clockcycleswait, d_time, N);
            TimeKernel<<<1,N>>>(*dd_time, N);
        }
        cudaDeviceSynchronize();
        chTimerGetTime( &stop );
        cudaMemcpy(&h_time, &d_time, sizeof(long long int), cudaMemcpyDeviceToHost);
        // cudaMemcpy(&hh_time, &dd_time, sizeof(long long int), cudaMemcpyDeviceToHost);
        
        printf( "count: %lld \n", h_time );
        microseconds = 1e6*chTimerElapsedTime( &start, &stop );
        usPerLaunch2Kernel = microseconds / (float) cIterations;
        
        printf( "%.5f us\n", usPerLaunch2Kernel );

    } while (usPerLaunch2Kernel <= 2*usPerLaunch);
        // printf( "%.5f us\n", usPerLaunch2Kernel );
           
    return 0;
}
