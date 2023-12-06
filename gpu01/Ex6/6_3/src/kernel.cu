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
__global__ void reduction_Kernel(int numElements, float* dataIn, float* dataOut)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (elementId < numElements)
	{
		shMem[blockIdx.y * tileSize + threadIdx.x] = matrixA[elementId];
        __synchthreads();
        for(int i = 0; i < tileSize; i++)
        {
            for(int j = 0; i < tileSize; j++)
            {
                sum += shMem[i * tileSize + j];
            }
        }
        __syncthreads();
        matrixRes[blockIdx.y * blockDim.y + blockIdx.x] = sum;
	}
}

void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, float* dataIn, float* dataOut) {
	reduction_Kernel<<< gridSize, blockSize>>>(numElements, dataIn, dataOut);
}
