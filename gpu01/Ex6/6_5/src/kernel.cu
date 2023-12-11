/**************************************************************************************************
 *
 *       Computer Engineering Group, Heidelberg University - GPU Computing Exercise 06
 *
 *                 Gruppe : TODO
 *
 *                   File : kernel.cu
 *
 *                Purpose : Reduction
 *
 **************************************************************************************************/

#include <stdio.h>
#include <device_launch_parameters.h>

#define FULL_MASK 0xffffffff

__global__ void opt_volt_reduction_Kernel(int numElements, int *dataIn, int *dataOut)
{
	extern __shared__ int sdata[];

	unsigned int warp_subset = blockDim.x > 32 ? 32 : blockDim.x;
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	// This ensures mapping 1:1 of all threads to underlying dataIn structure
	unsigned int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	int var = dataIn[elementId];
	for (unsigned int s = 1; s < warp_subset; s *= 2) {
		var += __shfl_xor_sync(FULL_MASK, var, s);
	}
	sdata[tid] = var;

	__syncthreads();

	// Do reduction of rest of the numbers shared mem
	for (unsigned int s = blockDim.x/2; s > 16 ; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}

	// write result for this block to global mem
	if (tid == 0) {
		dataOut[blockIdx.x] = sdata[0];
	}
}

void opt_volt_reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, int *dataIn, int *dataOut)
{
	int shared_mem = (numElements*sizeof(int))/gridSize.x;
	opt_volt_reduction_Kernel<<<gridSize, blockSize, shared_mem>>>(numElements, dataIn, dataOut);
}

