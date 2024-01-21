/*************************************************************************************************
 *
 *        Computer Engineering Group, Heidelberg University - GPU Computing Exercise 09
 *
 *                          Author : Lorenz Braun &  Hendrik Borras
 *
 *                            File : main.cpp
 *
 *                         Purpose : Stencil Code
 *
 *************************************************************************************************/

#include <iostream>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cmath>

const static int DEFAULT_NUM_ELEMENTS   = 1024;
const static int DEFAULT_NUM_ITERATIONS =    5;

//
// Structures
struct StencilArray_t {
    float* array;
    float* tmp_array;
    int    size; // size == width == height
};

//
// Function Prototypes
//
void printHelp(char *);
void writeToFile(float * matrix, const char * name, int size);

//
// Stencil Code Kernel for the cpu
//
void
cpu_simpleStencil(/* TODO Parameters */)
{    
    /* TODO Function code */
}

//
// Code for checking stability of solution on the cpu
//
float
cpu_calculate_max_diff(/* TODO Parameters */)
{    
    float max_diff = 0.;
    /* TODO Function code */
    return max_diff;
}


//
// Stencil Code Kernel for the gpu with acc
//
void
acc_simpleStencil(/* TODO Parameters */)
{    
    /* TODO Function code */
}

//
// Code for checking stability of solution on the gpu with acc
//
float
acc_calculate_max_diff(/* TODO Parameters */)
{    
    float max_diff = 0.;
    /* TODO Function code */
    return max_diff;
}

//
// Main
//
int
main(int argc, char * argv[])
{
    bool showHelp = chCommandLineGetBool("h", argc, argv);
    if (!showHelp) {
        showHelp = chCommandLineGetBool("help", argc, argv);
    }

    if (showHelp) {
        printHelp(argv[0]);
        exit(0);
    }

    std::cout << "***" << std::endl
              << "*** Starting ..." << std::endl
              << "***" << std::endl;

    ChTimer kernelTimer;

    //
    // Allocate Memory
    //
    int numElements = 0;
    chCommandLineGet<int>(&numElements, "s", argc, argv);
    chCommandLineGet<int>(&numElements, "size", argc, argv);
    numElements = numElements != 0 ? numElements : DEFAULT_NUM_ELEMENTS;

    //
    // Host Memory
    //
    
    // use acc kernel?
    bool useAcc = chCommandLineGetBool("acc", argc, argv);

    StencilArray_t h_array;
    h_array.size = numElements;
    // Pageable
    h_array.array = static_cast<float*> (malloc(static_cast<size_t> (h_array.size * h_array.size * sizeof(float))));
    h_array.tmp_array = static_cast<float*> (malloc(static_cast<size_t> (h_array.size * h_array.size * sizeof(float))));

    // Init patch
    bool noPatch = chCommandLineGetBool("noPatch", argc, argv);
    for (int j = 0; j < h_array.size; j++) {
        for ( int i = 0; i < h_array.size; i++) {
            // TODO: Initialize the array
            h_array.array[j + h_array.size*i] = 0;
            
            // copy over the initialization
            h_array.tmp_array[j + h_array.size*i] = h_array.array[j + h_array.size*i];
        }
    }
    
    // Visualize start state
    writeToFile(h_array.array, "init.pgm", h_array.size);

    if (h_array.array == NULL) {
        std::cout << "\033[31m***" << std::endl
                  << "*** Error - Memory allocation failed" << std::endl
                  << "***\033[0m" << std::endl;

        exit(-1);
    }
    

    //
    // Get Kernel Launch Parameters
    //
    int numIterations = 0;

    // Number of Iterations 
    chCommandLineGet<int>(&numIterations,"i", argc, argv);
    chCommandLineGet<int>(&numIterations,"num-iterations", argc, argv);
    numIterations = numIterations != 0 ? numIterations : DEFAULT_NUM_ITERATIONS;

    float curr_diff;
    kernelTimer.start();
    
    for (int i = 0; i < numIterations; i ++) {
        if (useAcc){
            acc_simpleStencil(/* TODO Parameters */);
            curr_diff = acc_calculate_max_diff(/* TODO Parameters */);
        } else {
            cpu_simpleStencil(/* TODO Parameters */);
            curr_diff = cpu_calculate_max_diff(/* TODO Parameters */);
        }
        
        // TODO Add exit criterion, when the solution stabilizes
        
        // Swap arrays
        float* tmp = h_array.array;
        h_array.array = h_array.tmp_array;
        h_array.tmp_array = tmp;
    }
    
    kernelTimer.stop();
    
    // Visualize result    
    writeToFile(h_array.array, "result.pgm", h_array.size);

    // Free Memory
    free(h_array.array);
    free(h_array.tmp_array);

    // Print Meassurement Results
    std::cout << "***" << std::endl
              << "*** Results:" << std::endl
              << "***    Size: " << numElements << std::endl
              << "***    Time for Stencil Computation: " << 1e3 * kernelTimer.getTime()
                << " ms" << std::endl
              << "***    Final maximum difference: " << curr_diff << std::endl
              << "***" << std::endl;

    return 0;
}

void
printHelp(char * argv)
{
    std::cout << "Help:" << std::endl
              << "  Usage: " << std::endl
              << "  " << argv << " [-p] [-s <num-elements>] [--acc] [--noPatch]"
                  << std::endl
              << "" << std::endl
              << "  -s <width-and-height>|--size <width-and-height>" << std::endl
              << "    Te width and the height of the array" << std::endl
              << "" << std::endl
              << "    The number of threads per block" << std::endl
              << "" << std::endl
              << "  --acc" 
                  << std::endl
              << "    Use the acc Kernel" << std::endl
              << "" << std::endl
              << "  --noPatch" 
                  << std::endl
              << "    Don't initalize grey patch in the middle of the matrix." << std::endl
              << "" << std::endl;

}

void writeToFile(float * grid, const char * name, int size)
{ 
    FILE * pFile;
    
    pFile = fopen (name,"w");
    int i,j;
    
    fprintf (pFile, "P2 %d %d %d\n", size, size, 127);
    
    for(i = 0; i<size; i++)
    {
        for(j = 0; j<size; j++)
        {
            fprintf (pFile, "%d ", (int) grid[j*size + i]);
        }
        fprintf (pFile, "\n");
    }
    
    fclose (pFile);
    
    return;
}
