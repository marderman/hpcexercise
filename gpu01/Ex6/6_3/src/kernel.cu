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
    //printf("elementId %d, stride %d, halfstride %d\n", elementId, stride, halfstride);
	if (elementId < numElements)
	{
        for(int i = 0; i < stride-1; i++){
            dataIn[elementId * stride] = dataIn[elementId * stride] + dataIn[(elementId * stride) + halfstride];
            __syncthreads();
        }
        //printf("%f ", dataIn[elementId * stride]);
        dataOut[0] = dataIn[elementId*stride];
        //printf("%d\n", dataOut[0]);
	}
}


void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float* dataIn, float* dataOut)
{
	reduction_Kernel<<< gridSize, blockSize>>>(numElements, dataIn, dataOut);
}
