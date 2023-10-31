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

__global__ void NullKernel() {}

int main (int argc, char *argv[])
{
    const int max_th_per_block = 1024;
    const int max_num_blocks = 16384;
    const int cIterations = 100;

    if (argc != 3) {
	fprintf (stderr, "Wrong number of arguments, there should be 2!\n");
	exit(1);
    }

    int act;
    int nb;
    int tpb;

    for ( int i = 1; i < argc; i++ ) {

	if (sscanf(argv[i], "%d", &act) != 1) {           //tests if input arguments are integers
	    fprintf( stderr, "Parameter %d is not an ineger!\n", i);
	    exit(EXIT_FAILURE);
	}

	act = atoi(argv[i]);

	if (i == 1) {
	    if (!(act > 0 && act < max_num_blocks)) {
		fprintf (stderr, "Wrong block number, the value should be between 1 and %d (exclusively)!\n", max_num_blocks);
		exit(1);
	    }
	    nb = act;
	} else if (i == 2) {
	    if (!(act > 0 && act < max_th_per_block)) {
		fprintf (stderr, "Wrong amount of threads per block, the value should be between 1 and %d (exclusively)!\n", max_th_per_block);
		exit(1);
	    }
	    tpb = act;
	}

    }

    double microseconds;
    double usPerLaunch;
    chTimerTimestamp start, stop;

    // printf( "SYNC - NB: %-5d | TPB: %-5d-> ", nb, tpb); fflush( stdout );
    printf("%d,%d", nb, tpb);

    chTimerGetTime( &start );

    for ( int i = 0; i < cIterations; i++ ) {
	NullKernel <<< nb, tpb>>>();
	cudaDeviceSynchronize();
    }

    chTimerGetTime( &stop );

    {
	microseconds = 1e6*chTimerElapsedTime( &start, &stop );
	usPerLaunch = microseconds / (float) cIterations;

	printf( ",%.2f", usPerLaunch );
    }

    chTimerGetTime( &start );

    for ( int i = 0; i < cIterations; i++ ) {
	NullKernel <<< nb, tpb>>>();
    }

    // Wait for all previous threads to complete
    cudaDeviceSynchronize();

    chTimerGetTime( &stop );

    {
	microseconds = 1e6*chTimerElapsedTime( &start, &stop );
	usPerLaunch = microseconds / (float) cIterations;

	printf( ",%.2f\n", usPerLaunch );
    }

    return 0;
}
