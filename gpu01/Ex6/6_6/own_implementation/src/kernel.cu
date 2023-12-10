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

//
// Reduction_Kernel
//
__global__ void
reduction_Kernel(int numElements, int *dataIn, int *dataOut)
{
	extern __shared__ int sPartials[];
    int sum = 0;
    const int tid = threadIdx.x;
    for ( size_t i = blockIdx.x*blockDim.x + tid;
          i < numElements;
          i += blockDim.x*gridDim.x ) {
        sum += dataIn[i];
		//printf("Block %d Thread %d accessing global at %d", blockIdx.x, tid, i);
    }
    sPartials[tid] = sum;
    __syncthreads();

    for ( int activeThreads = blockDim.x>>1; 
              activeThreads; 
              activeThreads >>= 1 ) {
        if ( tid < activeThreads ) {
            sPartials[tid] += sPartials[tid+activeThreads];
        }
        __syncthreads();
    }

    if ( tid == 0 ) {
        dataOut[blockIdx.x] = sPartials[0];
    }
}

void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, int *dataIn, int *dataOut)
{
	int shared_mem = blockSize.x*sizeof(int);
	reduction_Kernel<<<gridSize, blockSize, shared_mem>>>(numElements, dataIn, dataOut);
	
}
