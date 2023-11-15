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

#include <stdio.h>

//
// Test Kernel
//

__global__ void 
globalMem2SharedMem
//(/*TODO Parameters*/)
(float* d_mem, int elementsPerThread, int size)
{
	//Declare Shared Memory
	extern __shared__ float sharedMem[];

	//Thread Id on Block Grid used to determine which piece of the data should be copied
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//Start and End of Copied piece of data
	int startIdx = tid * elementsPerThread;
	//Start and End of Copied piece of data
	int endIdx = startIdx + elementsPerThread;
	

	//Kopiere die Daten in den Shared Memory
	for (int i = startIdx;i < endIdx;++i)
	{
		//printf("Copy Iteration %d\n", i);
		//Makes sure that if the amount is not equal distributed among the threads that the last thread will not copy too much data.
		if (i < size)
		{
			sharedMem[i] = d_mem[i];
		}
		else 
		{
			//printf("Thread doing nothing data element %d is out of range, too many threads for to less data\n", i);
		}
	}
		__syncthreads();
	d_mem[startIdx] = sharedMem[startIdx];

}


void globalMem2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* d_mem/* TODO Parameters*/) {
{
	//Calculate the elements each thread has to copy
	int elementsperThread = (shmSize/sizeof(float)) / ((gridSize.x * blockSize.x));
	//printf("Copying %d elements per thread\n", elementsperThread);
	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>(d_mem,elementsperThread,(shmSize/sizeof(float)));
}
}

__global__ void 
SharedMem2globalMem
//(/*TODO Parameters*/)
(float* d_mem, int elementsPerThread, int size )
{
	//Declare Shared Memory
	extern __shared__ float sharedMem[];

	//Thread Id on Block Grid used to determine which piece of the data should be copied
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//Start and End of Copied piece of data
	int startIdx = tid * elementsPerThread;
	//Start and End of Copied piece of data
	int endIdx = startIdx + elementsPerThread;
	

	//Kopiere die Daten in den Global Memory
	for (int i = startIdx;i < endIdx;++i)
	{
		//printf("Copy Iteration %d\n", i);
		//Makes sure that if the amount is not equal distributed among the threads that the last thread will not copy too much data.
		if (i < size)
		{
			d_mem[i] = sharedMem[i];
		}
		else 
		{
			//printf("Thread doing nothing data element %d is out of range, too many threads for to less data\n", i);
		}
	}
		__syncthreads();


}
void SharedMem2globalMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* d_mem  /* TODO Parameters*/) {
	//Calculate the elements each thread has to copy
	int elementsperThread = (shmSize / sizeof(int)) / ((gridSize.x * blockSize.x)-1);
	SharedMem2globalMem<<< gridSize, blockSize ,shmSize>>>(d_mem,elementsperThread,(shmSize/sizeof(float)));
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
