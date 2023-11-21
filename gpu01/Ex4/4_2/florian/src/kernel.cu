/******************************************************************************
 *
 *Computer Engineering Group, Heidelberg University - GPU Computing Exercise 04
 *
 *                  Group : TBD
 *
 *                   File : kernel.cu
 *
 *                Purpose : Memory Operations Benchmark
 *
 ******************************************************************************/


//
// Test Kernel
//
#include <stdio.h>
__global__ void globalMem2SharedMem(float* d_mem, float* dmem_B, int elementsPerThread, int elements)
{
	extern __shared__ float sharedMem[];
	int startadress = blockIdx.x * blockDim.x + threadIdx.x;
	if(startadress + elementsPerThread < elements - 1)
	{
		for(int i = startadress; i < startadress + elementsPerThread; i++)
		{
			sharedMem[i] = d_mem[i];
			// printf("i: %d\n", i);
			__syncthreads();
		}
		// printf("threadIdx.x: %d dmem_B[startadress]: %f", threadIdx.x, dmem_B[startadress]);
		dmem_B[startadress] = sharedMem[startadress];
	}
}

void globalMem2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* d_mem, float* dmem_B, int elements) {
	int elementsPerThread = elements / ((gridSize.x * blockSize.x) - 1);
	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>(d_mem, dmem_B, elementsPerThread, elements);
}

__global__ void SharedMem2globalMem(float* d_mem, float* dmem_B, int elementsPerThread, int elements)
{
	extern __shared__ float sharedMem[];
	int startadress = blockIdx.x * blockDim.x + threadIdx.x;
	if(startadress + elementsPerThread < elements - 1)
	{
		for(int i = startadress; i < startadress + elementsPerThread; i++)
		{
			sharedMem[i] = dmem_B[i];
			__syncthreads();
		}
		// printf("threadIdx.x: %d dmem_B[startadress]: %f", threadIdx.x, dmem_B[startadress]);
		// dmem_B[startadress] = sharedMem[startadress];
	}
}

void SharedMem2globalMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* d_mem, float* dmem_B, int elements) {
	int elementsPerThread = elements / ((gridSize.x * blockSize.x) - 1); 
	SharedMem2globalMem<<< gridSize, blockSize, shmSize >>>(d_mem, dmem_B, elementsPerThread, elements);
}

__global__ void 
SharedMem2Registers
//(/*TODO Parameters*/)
( )
{
	/*TODO Kernel Code*/
}
void SharedMem2Registers_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
	SharedMem2Registers<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}

__global__ void 
Registers2SharedMem
//(/*TODO Parameters*/)
( )
{
	/*TODO Kernel Code*/
}
void Registers2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
	Registers2SharedMem<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}

__global__ void 
bankConflictsRead
//(/*TODO Parameters*/)
( )
{
	/*TODO Kernel Code*/
}

void bankConflictsRead_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
	bankConflictsRead<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}
