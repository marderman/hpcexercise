/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 03
 *
 *                           Group : TBD
 *
 *                            File : main.cu
 *
 *                         Purpose : Memory Operations Benchmark
 *
 *************************************************************************************************/

//
// Kernels
//

__global__ void 
globalMemCoalescedKernel(int* dst, int* src, int elementsPerThread)
{
    int Start = threadIdx.x * blockIdx.x * elementsPerThread;
    for(int i = Start; i < Start + elementsPerThread; i++)
    {
        dst[i] = src[i];
    }
}

void 
globalMemCoalescedKernel_Wrapper(dim3 gridDim, dim3 blockDim, int* dst, int* src, int elements) {
    int elementsPerThread = elements / ((gridDim.x * blockDim.x) - 1);
	globalMemCoalescedKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>( dst, src, elementsPerThread);
}

__global__ void 
globalMemStrideKernel(int* dst, int* src, int stride )
{
    int i = (threadIdx.x + blockDim.x*blockIdx.x)*stride;
    dst[i] = src[i];
}

void 
globalMemStrideKernel_Wrapper(dim3 gridDim, dim3 blockDim, int* dst, int* src, int stride/*TODO Parameters*/) {
	globalMemStrideKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(dst, src, stride /*TODO Parameters*/);
}

__global__ void 
globalMemOffsetKernel(/*TODO Parameters*/)
{
    /*TODO Kernel Code*/
}

void 
globalMemOffsetKernel_Wrapper(dim3 gridDim, dim3 blockDim /*TODO Parameters*/) {
	globalMemOffsetKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>( /*TODO Parameters*/);
}

