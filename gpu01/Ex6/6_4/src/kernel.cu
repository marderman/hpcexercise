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
__global__ void reduction_Kernel(int numElements, float* dataIn, float* dataOut)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = numElements / (blockDim.x * gridDim.x);
    int halfstride = stride/2;
    __shared__ float value[1];
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
            __syncthreads();
        }
        else
        {   
            atomicAdd_system(value, dataIn[elementId]);
            // atomicAdd_system(value, dataIn[elementId * halfstride]);
            __syncthreads();
            *dataOut = value[0];
            __syncthreads();
        }
	}
}


void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float* dataIn, float* dataOut)
{
	reduction_Kernel<<< gridSize, blockSize>>>(numElements, dataIn, dataOut);
}
