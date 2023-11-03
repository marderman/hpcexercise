/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 03
 *
 *                           Group : 01
 *
 *                            File : main.cu
 *
 *                         Purpose : Memory Operations Benchmark
 *
 *************************************************************************************************/

//
// Kernels
//

#include <stdio.h>

__global__ void 
globalMemCoalescedKernel(int *src, int *dest,int elementCount, int elementsPerThread)
{
    int iStart = threadIdx.x + blockDim.x*blockIdx.x;
    int i;
    for (int j = 0; j < elementsPerThread; j++)
    {
        i = iStart + blockDim.x*gridDim.x*j;
        //printf("i: %d\n",i);
        if (i<elementCount){
            dest[i] = src[i];
        }
    }
}

void 
globalMemCoalescedKernel_Wrapper(dim3 gridDim, dim3 blockDim, int *src, int *dest,int elementCount) {
    int elementsPerThread = (elementCount + blockDim.x*gridDim.x - 1) / (blockDim.x*gridDim.x);
	globalMemCoalescedKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(src,dest,elementCount,elementsPerThread);
}

__global__ void 
globalMemStrideKernel(int *src, int *dest, int stride)
{
    int i = (threadIdx.x + blockDim.x*blockIdx.x)*stride;
    dest[i] = src[i];
}

void 
globalMemStrideKernel_Wrapper(dim3 gridDim, dim3 blockDim, int *src, int *dest, int stride) {
	globalMemStrideKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(src, dest, stride);
}

__global__ void 
globalMemOffsetKernel(int *src, int *dest, int offset)
{
    int i = (threadIdx.x + blockDim.x*blockIdx.x)+offset;
    dest[i] = src[i];
}

void 
globalMemOffsetKernel_Wrapper(dim3 gridDim, dim3 blockDim, int *src, int *dest, int offset) {
	globalMemOffsetKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(src, dest, offset);
}
