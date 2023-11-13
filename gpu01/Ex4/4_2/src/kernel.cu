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

__global__ void 
bankConflictsRead
//(/*TODO Parameters*/)
( )
{
	/*TODO Kernel Code*/
}

void bankConflictsRead_Wrapper(dim3 gridSize, dim3 blockSize, int shmSize /* TODO Parameters*/) {
	bankConflictsRead<<< gridSize, blockSize, shmSize >>>( /* TODO Parameters */);
}
