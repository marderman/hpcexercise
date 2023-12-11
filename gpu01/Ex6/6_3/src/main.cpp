/**************************************************************************************************
 *
 *       Computer Engineering Group, Heidelberg University - GPU Computing Exercise 06
 *
 *                 Gruppe : TODO
 *
 *                   File : main.cpp
 *
 *                Purpose : Reduction
 *
 **************************************************************************************************/

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cuda_runtime.h>
#include <stdio.h>

const static int DEFAULT_MATRIX_SIZE = 1024;
const static int DEFAULT_BLOCK_DIM   =  128;

//
// Function Prototypes
//
void printHelp(char *);
void printArray(int size, int *arr);


extern void reduction_Kernel_Wrapper(dim3 gridSize, dim3 blockSize, int numElements, int* dataIn, int* dataOut);

//
// Main
//
int
main(int argc, char * argv[])
{
	bool showHelp = chCommandLineGetBool("h", argc, argv);
	if (!showHelp)
	{
		showHelp = chCommandLineGetBool("help", argc, argv);
	}

	if (showHelp)
	{
		printHelp(argv[0]);
		exit(0);
	}

	std::cout << "***" << std::endl
			  << "*** Starting ..." << std::endl
			  << "***" << std::endl;

	ChTimer memCpyH2DTimer, memCpyD2HTimer;
	ChTimer kernelTimer;

	//
	// Allocate Memory
	//
	int numElements = 0;
	chCommandLineGet<int>(&numElements, "s", argc, argv);
	chCommandLineGet<int>(&numElements, "size", argc, argv);
	numElements = numElements != 0 ?
			numElements : DEFAULT_MATRIX_SIZE;
	//
	// Host Memory
	//
	bool pinnedMemory = chCommandLineGetBool("p", argc, argv);
	if (!pinnedMemory)
	{
		pinnedMemory = chCommandLineGetBool("pinned-memory",argc,argv);
	}

	int* h_dataIn = NULL;
	int* h_dataOut = NULL;
	if (!pinnedMemory)
	{
		// Pageable
		h_dataIn = static_cast<int*>
				(malloc(static_cast<size_t>(numElements * sizeof(*h_dataIn))));
		h_dataOut = static_cast<int*>
				(malloc(static_cast<size_t>(sizeof(*h_dataOut))));
	}
	else
	{
		// Pinned
		cudaMallocHost(&h_dataIn, 
				static_cast<size_t>(numElements * sizeof(*h_dataIn)));
		cudaMallocHost(&h_dataOut, 
				static_cast<size_t>(sizeof(*h_dataOut)));
	}
	// Init h_dataOut
	*h_dataOut = 0;

	// Device Memory
	int* d_dataIn = NULL;
	int* d_dataOut = NULL;
	int* d_partialSums;
	cudaMalloc(&d_dataIn, 
			static_cast<size_t>(numElements * sizeof(*d_dataIn)));
	cudaMalloc(&d_dataOut, 
			static_cast<size_t>(sizeof(*d_dataOut)));

	// Allocate device memory for partial sums
	cudaMalloc(&d_partialSums, sizeof(*d_dataOut));

	if (h_dataIn == NULL || h_dataOut == NULL ||
		d_dataIn == NULL || d_dataOut == NULL ||
		d_partialSums == NULL)
	{
		std::cout << "\033[31m***" << std::endl
		          << "*** Error - Memory allocation failed" << std::endl
		          << "***\033[0m" << std::endl;

		exit(-1);
	}

	
	//
    // Init Matrices
    //
    for (int i = 0; i < numElements; i++) {
   
        h_dataIn[i] = 1;
		// printf("%f ", h_dataIn[i]);
        // h_dataOut[i] = 0.0;
    }

	//printArray(numElements, h_dataIn);

	//
	// Copy Data to the Device
	//
	memCpyH2DTimer.start();

	cudaMemcpy(d_dataIn, h_dataIn, 
			static_cast<size_t>(numElements * sizeof(*d_dataIn)), 
			cudaMemcpyHostToDevice);
	cudaMemcpy(d_dataOut, h_dataOut, 
			static_cast<size_t>(sizeof(*d_dataOut)), 
			cudaMemcpyHostToDevice);

	memCpyH2DTimer.stop();

	//
	// Get Kernel Launch Parameters
	//
	int blockSize = 0,
		gridSize = 0;

	// Block Dimension / Threads per Block
	chCommandLineGet<int>(&blockSize,"t", argc, argv);
	chCommandLineGet<int>(&blockSize,"threads-per-block", argc, argv);
	blockSize = blockSize != 0 ?
			blockSize : DEFAULT_BLOCK_DIM;

	if (blockSize > 1024)
	{
		std::cout << "\033[31m***" << std::endl
		          << "*** Error - The number of threads per block is too big" << std::endl
		          << "***\033[0m" << std::endl;

		exit(-1);
	}

	gridSize = ceil(static_cast<int>(numElements) / static_cast<int>(blockSize));

	dim3 grid_dim = dim3(gridSize);
	dim3 block_dim = dim3(blockSize);

	kernelTimer.start();


	printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n", 
	grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z);

	fflush(stdout);

	//Do Partial Sums
	reduction_Kernel_Wrapper(grid_dim, block_dim, numElements, d_dataIn, d_partialSums);

	//Get Final Results
	reduction_Kernel_Wrapper(1, block_dim, numElements, d_dataIn, d_dataOut);

	// Synchronize
	cudaDeviceSynchronize();

	// Check for Errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		std::cout << "\033[31m***" << std::endl
				  << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
				  	<< std::endl
				  << "***\033[0m" << std::endl;

		return -1;
	}

	kernelTimer.stop();

	//
	// Copy Back Data
	//
	memCpyD2HTimer.start();

	cudaMemcpy(h_dataOut, d_dataOut, 
			static_cast<size_t>(sizeof(*d_dataOut)), 
			cudaMemcpyDeviceToHost);

	memCpyD2HTimer.stop();

	printf("Result: %d\n", h_dataOut[0]);

	//printArray(numElements, h_dataIn);

	// Free Memory
	if (!pinnedMemory)
	{
		free(h_dataIn);
		free(h_dataOut);
	}
	else
	{
		cudaFreeHost(h_dataIn);
		cudaFreeHost(h_dataOut);
	}
	cudaFree(d_dataIn);
	cudaFree(d_dataOut);
	cudaFree(d_partialSums);
	
	// Print Meassurement Results
	std::cout << "***" << std::endl
			  << "*** Results:" << std::endl
			  << "***    Num Elements: " << numElements << std::endl
			  << "***    Time to Copy to Device: " << 1e3 * memCpyH2DTimer.getTime()
			  	<< " ms" << std::endl
			  << "***    Copy Bandwidth: " 
			  	<< 1e-9 * memCpyH2DTimer.getBandwidth(numElements * sizeof(*h_dataIn))
			  	<< " GB/s" << std::endl
			  << "***    Time to Copy from Device: " << 1e3 * memCpyD2HTimer.getTime()
			  	<< " ms" << std::endl
			  << "***    Copy Bandwidth: " 
			  	<< 1e-9 * memCpyD2HTimer.getBandwidth(sizeof(*h_dataOut))
				<< " GB/s" << std::endl
			  << "***    Time for Reduction: " << 1e3 * kernelTimer.getTime()
				  << " ms" << std::endl
			  << "***" << std::endl;

	return 0;
}

void
printHelp(char * argv)
{
	std::cout << "Help:" << std::endl
			  << "  Usage: " << std::endl
			  << "  " << argv << " [-p] [-s <num-elements>] [-t <threads_per_block>]"
			  	<< std::endl
			  << "" << std::endl
			  << "  -p|--pinned-memory" << std::endl
			  << "	Use pinned Memory instead of pageable memory" << std::endl
			  << "" << std::endl
			  << "  -s <num-elements>|--size <num-elements>" << std::endl
			  << "	The size of the Matrix" << std::endl
			  << "" << std::endl
			  << "  -t <threads_per_block>|--threads-per-block <threads_per_block>" 
			  	<< std::endl
			  << "	The number of threads per block" << std::endl
			  << "" << std::endl;
}

void printArray(int size, int *arr)
{
    for(int i = 0; i < size; i++)
    {
        printf("%d ", arr[i]);
    }
	printf("\n\n");
}
