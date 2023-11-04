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
globalMemCoalescedKernel(int* source, int* destination, int size, int itemsperthread)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    for(int i = 0; i < itemsperthread; i++){
    
        if (tid >= size) return;

    destination[tid] = source[tid+i];
    }      
}

void 
globalMemCoalescedKernel_Wrapper(dim3 gridDim, dim3 blockDim, int* source, int* destination, int size) {
    int itemsperthread = (size / sizeof(int)) / (gridDim.x * blockDim.x);
	globalMemCoalescedKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(source, destination, size, itemsperthread);
}

__global__ void 
globalMemStrideKernel(int* source, int* destination, int size, int stride)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= size) return;

    destination[tid] = source[tid*stride]; 
}

void 
globalMemStrideKernel_Wrapper(dim3 gridDim, dim3 blockDim, int* source, int* destination, int size, int stride) {
	globalMemStrideKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(source, destination, size, stride);
}

__global__ void 
globalMemOffsetKernel(int* source, int* destination, int size, int offset)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= size) return;

    destination[tid] = source[tid*offset]; 
}

void 
globalMemOffsetKernel_Wrapper(dim3 gridDim, dim3 blockDim, int* source, int* destination, int size, int offset) {
	globalMemOffsetKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(source, destination, size, offset);
}