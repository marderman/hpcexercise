/**************************************************************************************************
 *
 *       Computer Engineering Group, Heidelberg University - GPU Computing Exercise 05
 *
 *                                 Group : GPU01
 *
 *                                  File : main.cu
 *
 *                               Purpose : Naive Matrix Multiplication
 *
 *************************************************************************************************/

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cuda_runtime.h>

const static int DEFAULT_MATRIX_WIDTH  = 1024;
const static int DEFAULT_BLOCK_DIM     =   32;

//
// Function Prototypes
//
void printHelp(char * /*programName*/);
void printOutMatrix(float *matrix, int width);
bool MatrixCompare(float* P, float* Q, long  matWidth);
//
// matMul_Kernel
//
__global__ void reduction_Kernel(int matrixSize, int tileSize, float *matrixA, float *matrixRes)
{
    extern __shared__ float shMem[];
    float sum = 0;
    int elementIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int elementIdy = blockIdx.y * blockDim.y + threadIdx.y;

    int elementId = elementIdy * matrixSize + elementIdx;

    if (elementIdx < matrixSize && 
        elementIdy < matrixSize)
    {
        shMem[threadIdx.y * tileSize + threadIdx.x] = matrixA[elementId];
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

//
// Main
//
int
main(int argc, char * argv[])
{
    //
    // Show Help
    //
    bool showHelp = chCommandLineGetBool("h", argc, argv);
    if (!showHelp) {
        showHelp = chCommandLineGetBool("help", argc, argv);
    }

    if (showHelp) {
        printHelp(argv[0]);
        exit(0);
    }

    ChTimer memCpyH2DTimer, memCpyD2HTimer;
    ChTimer kernelTimer;

    //
    // Allocate Memory
    //
    int matrixWidth = 0;
    chCommandLineGet<int>(&matrixWidth, "s", argc, argv);
    chCommandLineGet<int>(&matrixWidth, "size", argc, argv);
    matrixWidth = matrixWidth != 0 ? matrixWidth : DEFAULT_MATRIX_WIDTH;

    int matrixSize = matrixWidth * matrixWidth;

    //
    // Host Memory
    //
    bool pinnedMemory = chCommandLineGetBool("p", argc, argv);
    if (!pinnedMemory) {
        pinnedMemory = chCommandLineGetBool("pinned-memory",argc,argv);
    }

    float* h_matrixA = nullptr;
    float* h_matrixRes = nullptr;
    if (!pinnedMemory) {
        // Pageable
        h_matrixA = static_cast<float*>(malloc(static_cast<size_t>(matrixSize * sizeof(*h_matrixA))));
        h_matrixRes = static_cast<float*>(calloc(static_cast<size_t>(matrixSize), sizeof(*h_matrixRes)));

        if (!h_matrixA || !h_matrixRes) {
            printf("Error: Unable to allocate pageable memory\n");
            exit(EXIT_FAILURE);
        }

    } else 
    {
        // Pinned
        cudaError_t error11 =  cudaMallocHost(&h_matrixA, static_cast<size_t>(matrixSize * sizeof(*h_matrixA)));
        cudaError_t error13 = cudaMallocHost(&h_matrixRes, static_cast<size_t>(matrixSize * sizeof(*h_matrixRes)));
        memset ( h_matrixRes, 0, matrixSize * sizeof(*h_matrixRes) );

        if ( error11 != cudaSuccess || error13 != cudaSuccess) 
        {
            printf ( "cudaMemcpy failed: %s\n", cudaGetErrorString ( error11) );
        }
    }

    //
    // Device Memory
    //
    float* d_matrixA = nullptr;
    float* d_matrixRes = nullptr;
    cudaError_t error1 = cudaMalloc(&d_matrixA, static_cast<size_t>(matrixSize * sizeof(d_matrixA)));
    cudaError_t error3 = cudaMalloc(&d_matrixRes, static_cast<size_t>(matrixSize * sizeof(*d_matrixRes)));

    if ( error1 != cudaSuccess || error3 != cudaSuccess) 
    {
        printf ( "cudaMemcpy failed: %s\n", cudaGetErrorString ( error1 ) );
    }


    //
    // Check Pointers
    //
    if (h_matrixA == NULL ||  h_matrixRes == NULL ||
        d_matrixA == NULL || d_matrixRes == NULL )
    {
        std::cout << "\033[31m***" << std::endl
                  << "*** Error - Allocation of Memory failed!!!" << std::endl
                  << "***\033[0m" << std::endl;
        exit(-1);
    }

    //
    // Init Matrices
    //
    for (int i = 0; i < matrixSize; i++) {
        int x = i % matrixWidth;
        int y = i / matrixWidth;
        h_matrixA[i] = static_cast<float>(x * y);
    }

    //
    // Copy Data to the Device
    //
    memCpyH2DTimer.start();

    cudaMemcpy(d_matrixA, h_matrixA, static_cast<size_t>(matrixSize * sizeof(*d_matrixA)), 
            cudaMemcpyHostToDevice);

    memCpyH2DTimer.stop();

    //
    // Get Kernel Launch Parameters
    //
    int tileWidth = 0, tileSize = 0, gridSize = 0;

    // Block Dimension / Threads per Block
    // chCommandLineGet<int>(&blockSize,"t", argc, argv);
    chCommandLineGet<int>(&tileWidth,"tilewidth", argc, argv);

    if(matrixSize % tileWidth != 0)
    {
        std::cout << "error matrixSize '%' tileWidth != 0" << std::endl;
    }
    else
    {
        gridSize = ceil(static_cast<float>(matrixWidth) / static_cast<float>(tileWidth));
        tileSize = tileWidth * tileWidth;
    }


    dim3 grid_dim = dim3(gridSize, gridSize, 1);
    dim3 block_dim = dim3(tileSize, tileSize, 1);

    /*std::cout << "***" << std::endl
              << "*** Grid Dim:  " << grid_dim.x << "x" << grid_dim.y << "x" << grid_dim.z 
                      << std::endl
              << "*** Block Dim: " << block_dim.x << "x" << block_dim.y << "x" << block_dim.z 
                      << std::endl
              << "***" << std::endl;*/

    // TODO Calc shared mem size
    int sharedMemSize = tileSize * sizeof(float);

    kernelTimer.start();

    //
    // Launch Kernel
    //
    if (chCommandLineGetBool("shared", argc, argv)) {
        // shMatMul_Kernel<<<grid_dim, block_dim, sharedMemSize>>>(matrixWidth, d_matrixA, d_matrixRes);   
        reduction_Kernel<<<grid_dim, block_dim, sharedMemSize>>>(matrixWidth, tileSize, d_matrixA, d_matrixRes);

    } else {
        // matMul_Kernel<<<grid_dim, block_dim>>>(matrixWidth, d_matrixA, d_matrixRes);
    }

    //
    // Synchronize
    //
    cudaDeviceSynchronize();

    //
    // Check for Errors
    //
    cudaError_t cudaError = cudaGetLastError();
    if ( cudaError != cudaSuccess ) {
        std::cout << "\033[***" << std::endl
                  << "***ERROR3*** " << cudaError << " - " << cudaGetErrorString(cudaError)
                    << std::endl
                  << "***\033[0m" << std::endl;

        return -1;
    }

    kernelTimer.stop();

    //
    // Copy Back Data
    //
    memCpyD2HTimer.start();

    cudaMemcpy(h_matrixRes, d_matrixRes, static_cast<size_t>(matrixSize * sizeof(*d_matrixRes)), 
            cudaMemcpyDeviceToHost);

    memCpyD2HTimer.stop();

    //
    // Check Result
    //
    bool dontCheckResult = chCommandLineGetBool("c", argc, argv);
    if (!dontCheckResult) {
        dontCheckResult = chCommandLineGetBool("no-check", argc, argv);
    }

    if (!dontCheckResult) {
        float* h_matrixD = static_cast<float*>(
                calloc(static_cast<size_t>(matrixSize), sizeof(*h_matrixD)));

        // MatrixMulOnHostBlocked(h_matrixA, h_matrixB, h_matrixD, 
        //         static_cast<long>(matrixWidth), 32);

        bool resultOk = MatrixCompare(h_matrixRes, h_matrixD, 
                static_cast<long>(matrixWidth));

        if (!resultOk) {
            std::cout << "\033[31m***" << std::endl
                      << "*** Error - The two matrices are different!!!" << std::endl
                      << "***\033[0m" << std::endl;

            exit(-1);
        }

        free(h_matrixD);
    }

    //
    // Print Meassurement Results
    //
    std::cout       << "Grid Dim;" << grid_dim.x 
                    << ";Matrix Size;" << matrixSize
                    << ";Time to Copy to Device;" << 1e3 * memCpyH2DTimer.getTime()
                    << ";ms"
                    << ";Copy Bandwidth;" 
                    << 1e-9 * memCpyH2DTimer.getBandwidth(2 * matrixSize * sizeof(h_matrixA))
                    << ";GB/s;"
                    << "Time to Copy from Device;" << 1e3 * memCpyD2HTimer.getTime()
                    << ";ms;"
                    << "Copy Bandwidth;" 
                    << 1e-9 * memCpyD2HTimer.getBandwidth(matrixSize * sizeof(h_matrixA))
                    << ";GB/s;"
                    << "Time for Matrix Multiplication;" << 1e3 * kernelTimer.getTime()
                    << ";ms" << std::endl;

    if (chCommandLineGetBool("print-matrix", argc, argv) 
       && matrixWidth <= 16) {
        printOutMatrix(h_matrixRes, matrixWidth);
    }

    // Free Memory
    if (!pinnedMemory) {
        free(h_matrixA);
        free(h_matrixRes);
    } else {
        cudaFreeHost(h_matrixA);
        cudaFreeHost(h_matrixRes);
    }
    cudaFree(d_matrixA);
    cudaFree(d_matrixRes);

    return 0;
}

void printHelp(char * programName)
{
    std::cout << "Help:" << std::endl
              << "  Usage: " << std::endl
              << "  " << programName << " [-p] [-s <matrix_size>] [-t <threads_per_block>]" 
                << std::endl
              << "                 [-g <blocks_per_grid] [-c] [--print-matrix]" 
                << std::endl
              << "" << std::endl
              << "  -p|--pinned-memory" << std::endl
              << "  Use pinned Memory instead of pageable memory" << std::endl
              << "" << std::endl
              << "  -s <matrix_size>|--size <matix_size>" << std::endl
              << "  The width of the Matrix" << std::endl
              << "" << std::endl
              << "  -t <threads_per_block>|--threads-per-block <threads_per_block>" 
                << std::endl
              << "  The number of threads per block" << std::endl
              << "" << std::endl
              << "  -c|--no-checking" << std::endl
              << "  Do not check the result of the matrix multiplication" << std::endl
              << "" << std::endl
              << "  --print-matrix" << std::endl
              << "  Print the output matrix (only recommended for small matrices)" << std::endl
              << std::endl;
}

void printOutMatrix(float *matrix, int width) 
{
	int i;
	for (i = 0; i < width*width; i++) {
		printf ("%4.2f\t", matrix[i%width + (i/width) * width]);
		if ((i+1) % width == 0) printf ("\n");
		}
	printf ("\n");
}

bool MatrixCompare(float* P, float* Q, long  matWidth)
{
	long i;

	for ( i = 0; i < matWidth * matWidth; i++ ) {
		//if ( P[i] != Q[i] )
		// Holger 09.04.2014 floating point calculations might have small errors depending on the operation order
		if ( fabs ( ( P[i]-Q[i] ) / ( P[i]+Q[i] ) ) > 1E-05 )
			return false;
	}
	return true;
}
