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

__global__ void globalMemCoalescedKernel(int* dest, int* src, size_t nbytes) {

    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < nbytes/sizeof(int); i += blockDim.x*gridDim.x) {
	dest[i] = src[i];
    }
}

void globalMemCoalescedKernel_Wrapper(dim3 gridDim, dim3 blockDim, int* dest, int* src, size_t nbytes) {

    // Amount of data computed by each thread
    globalMemCoalescedKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>(dest,src,nbytes);
}

__global__ void globalMemStrideKernel(/*TODO Parameters*/) {
    /*TODO Kernel Code*/
}

void globalMemStrideKernel_Wrapper(dim3 gridDim, dim3 blockDim /*TODO Parameters*/) {
	globalMemStrideKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>( /*TODO Parameters*/);
}

__global__ void globalMemOffsetKernel(/*TODO Parameters*/) {
    /*TODO Kernel Code*/
}

void globalMemOffsetKernel_Wrapper(dim3 gridDim, dim3 blockDim /*TODO Parameters*/) {
	globalMemOffsetKernel<<< gridDim, blockDim, 0 /*Shared Memory Size*/ >>>( /*TODO Parameters*/);
}

