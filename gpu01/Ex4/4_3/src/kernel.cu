/******************************************************************************
 *
 *Computer Engineering Group, Heidelberg University - GPU Computing Exercise 04
 *
 *                  Group : TBD
 *
 *                   File : kernel.cu
 *
 *                Purpose : Memory Operations Benchmark
 *
 ******************************************************************************/


//
// Test Kernel
//

__global__ void 
globalMem2SharedMem
//(/*TODO Parameters*/)
( )
{
	/*TODO Kernel Code*/
}

void globalMem2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
	globalMem2SharedMem<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}

__global__ void 
SharedMem2globalMem
//(/*TODO Parameters*/)
( )
{
	/*TODO Kernel Code*/
}
void SharedMem2globalMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
	SharedMem2globalMem<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}

__global__ void 
SharedMem2Registers
//(/*TODO Parameters*/)
( )
{
	/*TODO Kernel Code*/
}
void SharedMem2Registers_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
	SharedMem2Registers<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}

__global__ void 
Registers2SharedMem
//(/*TODO Parameters*/)
( )
{
	/*TODO Kernel Code*/
}
void Registers2SharedMem_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
	Registers2SharedMem<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}

__global__ void bankConflictsRead(float* dummy, const int shmSize, int stride, long long int* clocks)
{
	long long int begin_cycles = clock64();
	extern __shared__ float s[];

	float t_reg = 0;

	 __syncthreads();

	// // Each thread loads data from global memory to shared memory
	// // if (globalIdx < size/sizeof(float)) {
	t_reg = s[threadIdx.x * stride];
	// // }

	long long int end_cycles = clock64();

	*dummy = t_reg;
	*clocks = end_cycles - begin_cycles;
}

void bankConflictsRead_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize, float* dummy, int stride, long long int* clocks)
{
	bankConflictsRead<<< gridSize, blockSize, shmSize >>>(dummy, shmSize, stride, clocks);
}
