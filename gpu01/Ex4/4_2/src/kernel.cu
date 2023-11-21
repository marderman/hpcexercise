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
#include <curand.h>

//
// Test Kernel
//

__global__ void 
globalMem2SharedMem
//(/*TODO Parameters*/)
(float* d_mem_a, int elementsPerThread, int size, float* d_mem_b)
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
			sharedMem[i] = d_mem_a[i];
		}
		else 
		{
			//printf("Thread doing nothing data element %d is out of range, too many threads for to less data\n", i);
		}
	}
		__syncthreads();
	d_mem_b[startIdx] = sharedMem[startIdx];

}


void globalMem2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* d_mem_a ,float* d_mem_b/* TODO Parameters*/) {
{
	//Calculate the elements each thread has to copy
	int elementsperThread = (shmSize/sizeof(float)) / ((gridSize.x * blockSize.x)-1);
	//printf("Copying %d elements per thread\n", elementsperThread);
	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>(d_mem_a,elementsperThread,(shmSize/sizeof(float)), d_mem_b);
}
}

__global__ void 
SharedMem2globalMem
//(/*TODO Parameters*/)
(float* d_mem, int elementsPerThread, int size, float* d_mem_b )
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
	}
		__syncthreads();


}
void SharedMem2globalMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* d_mem_a, float* d_mem_b  /* TODO Parameters*/) {
	//Calculate the elements each thread has to copy
	int elementsperThread = (shmSize / sizeof(int)) / ((gridSize.x * blockSize.x)-1);
	SharedMem2globalMem<<< gridSize, blockSize ,shmSize>>>(d_mem_a,elementsperThread,(shmSize/sizeof(float)),d_mem_b);
}

__global__ void 
SharedMem2Registers
//(/*TODO Parameters*/)
(float* d_mem, int elementsPerThread, int size)
{
	/*TODO Kernel Code*/
	//Declare Shared Memory
	extern __shared__ float sharedMem[];

	//Declare storage for Thread 64k for each block and 32 Warp means maximum register space per thread by 2048
	float temp[48*1024/sizeof(float)];
	//float temp[100];

	//Thread Id on Block Grid used to determine which piece of the data should be copied
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//Start and End of Copied piece of data
	int startIdx = tid * elementsPerThread;
	//Start and End of Copied piece of data
	int endIdx = startIdx + elementsPerThread;

	for (int i = startIdx, j = 0; i < endIdx;++i,++j)
	{
		if (i < size)
		{
		temp[j] = sharedMem[i];
		}
	}
	__syncthreads();
	sharedMem[startIdx] = temp[0];

}
void SharedMem2Registers_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* d_mem_a /* TODO Parameters*/) {
	int elementsperThread = (shmSize / sizeof(float)) / ((gridSize.x * blockSize.x)-1);
	//printf("Elements per Thread copied: %d\n", elementsperThread);
	//fflush(stdout);
	SharedMem2Registers<<< gridSize, blockSize, shmSize >>>(d_mem_a,elementsperThread, shmSize/sizeof(float));
}

__global__ void 
Registers2SharedMem
//(/*TODO Parameters*/)
(float* d_mem, int elementsPerThread, int size)
{
	/*TODO Kernel Code*/
	//Declare Shared Memory
	extern __shared__ float sharedMem[];

	//Declare storage for Thread 64k for each block and 32 Warp means maximum register space per thread by 2048
	
	float temp[48*1024/sizeof(float)];
	//float temp[100];

	//Thread Id on Block Grid used to determine which piece of the data should be copied
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	//Start and End of Copied piece of data
	int startIdx = tid * elementsPerThread;
	//Start and End of Copied piece of data
	int endIdx = startIdx + elementsPerThread;

	for (int i = startIdx, j = 0; i < endIdx;++i,++j)
	{
		if (i < size)
		{
		sharedMem[i] = temp[j];
		}
	}
	__syncthreads();
}
void Registers2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* d_mem_a/* TODO Parameters*/) {
	int elementsperThread = (shmSize / sizeof(float)) / ((gridSize.x * blockSize.x)-1);
	Registers2SharedMem<<< gridSize, blockSize, shmSize >>>(d_mem_a, elementsperThread, shmSize/sizeof(float));
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
