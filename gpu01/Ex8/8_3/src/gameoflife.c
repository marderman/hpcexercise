#include <stdio.h>
#include <stdlib.h>
#include <chTimer.hpp>
#include <nvToolsExt.h>
 
#define SRAND_VALUE 1985
 
//#define dim 32 // Linear game grid dimension excluding ghost cells
int dim;
#define idx(i,j) ((i)*(dim+2)+(j))
int vectorLength;
int numGangs;

inline void writeGridToStdout(int* grid, int iteration, int maxIter) 
{

    //fprintf(file, "Iteration %d:\n", iteration);

    for (int i = 1; i <= dim; i++) {
        for (int j = 1; j <= dim; j++) {
            printf("%d ", grid[idx(i,j)]);
        }
        printf("\n");

    }
    if (iteration != (maxIter-1))
    {
        printf("\n");
    }


}



inline int getNeighbors(int *grid, int i, int j)
{
    int num = grid[idx(i+1,j)] + grid[idx(i-1,j)]     //upper lower
            + grid[idx(i,j+1)] + grid[idx(i,j-1)]     //right left
            + grid[idx(i+1,j+1)] + grid[idx(i-1,j-1)] //diagonals
            + grid[idx(i-1,j+1)] + grid[idx(i+1,j-1)];
 
    return num;
}
 
int main(int argc, char* argv[])
{
    int i,j,iter;

    ChTimer gameTimer;

    dim = atoi(argv[1]);

    vectorLength = atoi(argv[2]);

    numGangs = atoi(argv[3]);
    // Number of game iterations
    int maxIter = 1<<10;
 
    // Total number of alive cells
    int total = 0;
 
    // Total size of grid in bytes
    size_t bytes = sizeof(int)*(dim+2)*(dim+2);
 
    // Allocate square grid of (dim+2)^2 elements, 2 added for ghost cells
    int *grid = (int*) malloc(bytes);
 
    // Allocate newGrid
    int *restrict newGrid = (int*) malloc(bytes);
 
    // Assign initial population randomly
    srand(SRAND_VALUE);
    for(i = 1; i <= dim; i++) {
        for(j = 1; j <= dim; j++) {
            grid[idx(i,j)] = rand() % 2;
        }
    }
 
  int fullSize = (dim+2)*(dim+2);
  #pragma acc data copyin(grid[0:fullSize]) create(newGrid[0:fullSize]) cache(grid[0:fullSize]) cache(newGrid[0:fullSize])
  {
    // Main game loop
    for (iter = 0; iter<maxIter; iter++) {
    gameTimer.start();
        
    #pragma acc kernels vector_length(vectorLength) num_gangs(numGangs)
    {
        // Left-Right columns
        #pragma acc loop independent 
        for (i = 1; i <= dim; i++) {
            grid[idx(i,0)] = grid[idx(i,dim)];   //Copy first real column to right ghost column
            grid[idx(i,dim+1)] = grid[idx(i,1)]; //Copy last real column to left ghost column
        }
        // Top-Bottom rows
        #pragma acc loop independent
        for (j = 0; j <= dim+1; j++) {
            grid[idx(0,j)] = grid[idx(dim,j)];   //Copy first real row to bottom ghost row
            grid[idx(dim+1,j)] = grid[idx(1,j)]; //Copy last real row to top ghost row
        }
 
        // Now we loop over all cells and determine their fate
        #pragma acc loop independent
        for (i = 1; i <= dim; i++) {
            for (j = 1; j <= dim; j++) {
               // Get the number of neighbors for a given grid point
                int numNeighbors = getNeighbors(grid,i,j);
 
                // Here we have explicitly all of the game rules
                if (grid[idx(i,j)] == 1 && numNeighbors < 2)
                    newGrid[idx(i,j)] = 0;
                else if (grid[idx(i,j)] == 1 && (numNeighbors == 2 || numNeighbors == 3))
                    newGrid[idx(i,j)] = 1;
                else if (grid[idx(i,j)] == 1 && numNeighbors > 3)
                    newGrid[idx(i,j)] = 0;
                else if (grid[idx(i,j)] == 0 && numNeighbors == 3)
                    newGrid[idx(i,j)] = 1;
                else
                    newGrid[idx(i,j)] = grid[idx(i,j)];
            }
        }
 
       // Can't switch pointers so we mannually have to copy array over
       #pragma acc loop independent 
       for(i = 1; i <= dim; i++) {
           for(j = 1; j <= dim; j++) {
               grid[idx(i,j)] = newGrid[idx(i,j)];
           }

       }
        //#pragma acc wait(iter)
        //writeGridToStdout(grid, iter, maxIter);
    } // End ACC kernels region
    } // End main game loop



    gameTimer.stop();
    // Sum up alive cells
    #pragma acc parallel
    {
	#pragma acc loop reduction(+:total)
        for (i = 1; i <= dim; i++) {
	    #pragma acc loop reduction(+:total)
            for (j = 1; j <= dim; j++) {
                total += grid[idx(i,j)];
            }
        }
    }
  } // End ACC Data region
    printf("%d;%d;%d;%d;%lf\n",vectorLength, numGangs, dim,total, gameTimer.getTime());
    fflush(stdout);
 
    // Release memory
    free(grid);
    free(newGrid);
 
    return 0;
}