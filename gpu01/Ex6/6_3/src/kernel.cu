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
//
// Reduction_Kernel
//
__global__ void reduction_Kernel(int numElements, float* dataIn, float* dataOut)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = numElements / (blockDim.x * gridDim.x);
    int halfstride = stride/2;
    float value = 0;
    // printf("elementId %d, stride %d, halfstride %d\n");
	if (elementId < numElements)
	{
        value = dataIn[elementId * stride] + dataIn[(elementId * stride) + halfstride];
        __syncthreads();
        dataIn[elementId * stride] = value;
        printf("%f ", dataIn[elementId * stride]);
	}
}


void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float* dataIn, float* dataOut)
{
	reduction_Kernel<<< gridSize, blockSize>>>(numElements, dataIn, dataOut);
}
