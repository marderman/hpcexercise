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



__global__ void TimeKernel(long long int* endtime, int* n)
{
    // endtime = 0;
    int i = threadIdx.x;
    if (i < *n)
    {
        long long int start = clock64();

        *endtime = 1000;
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


    long long int* h_time;
    long long int* h_waited_time;
    long long int* h_clockcycleswait;
    int* N;

    long long int* d_time;
    long long int* d_waited_time;
    long long int* d_clockcycleswait;

    h_time = (long long int*)calloc(1,sizeof(long long int));
    h_clockcycleswait = (long long int*)calloc(1,sizeof(long long int));
    h_waited_time = (long long int*)calloc(1,sizeof(long long int));

    N = (int*) malloc(1*sizeof(int));
    *N = 1000;


    cudaMalloc(&d_time, sizeof(long long int));
    cudaMalloc(&d_waited_time, sizeof(long long int));
    cudaMalloc(&d_clockcycleswait, sizeof(long long int));

    cudaMemcpy(d_time, h_time, sizeof(long long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_waited_time, h_waited_time, sizeof(long long int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_clockcycleswait, h_clockcycleswait, sizeof(long long int), cudaMemcpyHostToDevice);

    const int cIterations = 1000;
    printf( "Measuring asynchronous launch time... " ); fflush( stdout );

    chTimerTimestamp start, stop;

    chTimerGetTime( &start );
    for ( int i = 0; i < cIterations; i++ ) 
    {
        TimeKernel<<<1000,1>>>(d_waited_time, N);
    }

    cudaDeviceSynchronize();
    chTimerGetTime( &stop );
    cudaMemcpy(h_waited_time, d_waited_time, sizeof(long long int), cudaMemcpyDeviceToHost);
    printf("h_cycle_count: %lld ", *h_waited_time);

    double microseconds = 1e6*chTimerElapsedTime( &start, &stop );
    double usPerLaunch = microseconds / (float) cIterations;
  
    printf( "Needed Time: %.5f us\n", usPerLaunch ); fflush(stdout);
  
    // double usPerLaunch2Kernel = 0;
    
    // do
    // {
    //     *h_clockcycleswait += 100;
    //     printf("h_clockcycleswait: %lld ", *h_clockcycleswait);
    //     cudaMemcpy(d_clockcycleswait, h_clockcycleswait, sizeof(long long int), cudaMemcpyHostToDevice);

    //     chTimerGetTime( &start );
    //     for ( int i = 0; i < cIterations; i++ ) 
    //     {
    //         ActiveWaitKernel<<<1,*N>>>(*d_clockcycleswait, d_time, *N);
    //         // ActiveWaitKernel(*d_clockcycleswait, d_time, N);
    //     }
    //     cudaDeviceSynchronize();
    //     chTimerGetTime( &stop );
    //     cudaMemcpy(h_time, d_time, sizeof(long long int), cudaMemcpyDeviceToHost);
    //     //cudaMemcpy(hh_time, &dd_time, sizeof(long long int), cudaMemcpyDeviceToHost);
        
    //     printf( "count: %lld \n", *h_time );
    //     microseconds = 1e6*chTimerElapsedTime( &start, &stop );
    //     usPerLaunch2Kernel = microseconds / (float) cIterations;
        
    //     printf( "%.5f us\n", usPerLaunch2Kernel );

    // } while (usPerLaunch2Kernel <= 2*usPerLaunch);
    //     printf( "%.5f us\n", usPerLaunch2Kernel );
           
    free(h_clockcycleswait);
    free(h_time);
    free(h_waited_time);
    return 0;
}
