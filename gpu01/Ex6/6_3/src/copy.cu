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
reduction_Kernel(int numElements, float *dataIn, float *dataOut)
{
	extern __shared__ float sdata[];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int elementId = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = dataIn[elementId];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2)
	{
		if (tid % (2 * s) == 0)
		{
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global mem
	if (tid == 0){
		dataOut[blockIdx.x] = sdata[0];
	}
}

void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float *dataIn, float *dataOut)
{
	int shared_mem = (numElements*sizeof(float))/gridSize.x;
	reduction_Kernel<<<gridSize, blockSize, shared_mem>>>(numElements, dataIn, dataOut);
}
