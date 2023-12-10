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

    // printf("gridDim %d,blockdim %d\n", gridDim.x, blockDim.x);

	if (elementId < numElements)
	{
        // value = dataIn[elementId * stride] + dataIn[(elementId * stride) + halfstride];
        // __syncthreads();
        // dataIn[elementId * stride] = value;
        // // printf("elementid: %d, data[in]: %f \n",elementId*stride, dataIn[elementId * stride]);
        // if(gridDim.x == 1)
        // {
        //     for(int i = 0; i < blockDim.x; i=i+2)
        //     dataOut = 
        // }

        if(gridDim.x > 1)
        {
            atomicAdd_system(value, dataIn[elementId * stride]);
            __syncthreads();
            atomicAdd_system(value, dataIn[(elementId * stride) + halfstride]);
            __syncthreads();
            // dataIn[elementId * stride] = value[0];
            // __syncthreads();
            //  dataIn[elementId * stride+halfstride] = value[0];
            //  __syncthreads();

            if(threadIdx.x == 0)
            {
                // printf("thread.x 0 elementid: %d\n", elementId*stride);
                dataIn[elementId * stride] = value[0];
                // dataIn[elementId * stride] = 2.0;
            }
            // dataIn[elementId * stride] = 2.0;
            // dataIn[(elementId * stride) + halfstride] = 3.0;
            // __syncthreads();
            // printf("1 numElements %d,elementId %d, stride %d, halfstride %d, gridDim %d, value: %f, elementId * stride: %d, elementId * stride+halfstride: %d\n", numElements, elementId, stride, halfstride, gridDim.x, value[0], elementId * stride, elementId * stride+halfstride);
        }
        else
        {
            // printf("2 value: %f ", value[0]);
             __syncthreads();
            atomicAdd_system(value, dataIn[elementId * stride]);
            __syncthreads();
            // dataIn[blockDim.x] = value[0];
            *dataOut = value[0];
            // printf("2 numElements %d,elementId %d, stride %d, halfstride %d, gridDim %d, value: %f, elementId * stride: %d, elementId * stride+halfstride: %d\n", numElements, elementId, stride, halfstride, gridDim.x, value[0], elementId * stride, elementId * stride+halfstride);
        }

        // __syncthreads();
        // dataIn[elementId * stride] = value[0];
        // dataIn[(elementId * stride) + halfstride] = value[0];
        // __syncthreads();
        // value[0] = dataIn[elementId * stride] + dataIn[(elementId * stride) + halfstride];
        // *dataOut = value[0];
        // printf("value: %f ", value[0]);
	}
}


void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float* dataIn, float* dataOut)
{
	reduction_Kernel<<< gridSize, blockSize>>>(numElements, dataIn, dataOut);
}
