/**************************************************************************************************
 *
 *       Computer Engineering Group, Heidelberg University - GPU Computing Exercise 06
 *
 *                 Gruppe : 01
 *
 *                   File : kernel.cu
 *
 *                Purpose : Reduction
 *
 **************************************************************************************************/
#include <stdio.h>
//
// Reduction_Kernel
//
__global__ void reduction_Kernel(int numElements, int* dataIn, int* dataOut)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = numElements / (blockDim.x * gridDim.x);
    int halfstride = stride/2;
    __shared__ int value[1];
    value[0] = 0;

	if (elementId < numElements)
	{

        if(gridDim.x > 1)
        {
            atomicAdd_system(value, dataIn[elementId * 2]);
            __syncthreads();
            atomicAdd_system(value, dataIn[(elementId * 2) + 1]);

            if(threadIdx.x == 0)
            {
                __syncthreads();
                dataIn[blockIdx.x] = value[0];
            }
        }
        else
        {   
            atomicAdd_system(value, dataIn[elementId]);
            __syncthreads();
            *dataOut = value[0];
        }
	}
}


void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, int* dataIn, int* dataOut)
{
	reduction_Kernel<<< gridSize, blockSize>>>(numElements, dataIn, dataOut);
}
