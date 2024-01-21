#include <stdio.h>
#include <stdlib.h>
#include <chTimer.hpp>

#define SRAND_VALUE 1985

//#define dim 1024 // Linear game grid dimension excluding ghost cells
int dim;
#define idx(i, j) ((i) * (dim + 2) + (j))

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


// Add up all neighbors
inline int getNeighbors(int *grid, int i, int j)
{
    int num = grid[idx(i + 1, j)] + grid[idx(i - 1, j)]           // upper lower
              + grid[idx(i, j + 1)] + grid[idx(i, j - 1)]         // right left
              + grid[idx(i + 1, j + 1)] + grid[idx(i - 1, j - 1)] // diagonals
              + grid[idx(i - 1, j + 1)] + grid[idx(i + 1, j - 1)];

    return num;
}

int main(int argc, char *argv[])
{

    ChTimer gameTimer;


    dim = atoi(argv[1]);
    int i, j, iter;
    // Number of game iterations
    int maxIter = 1 << 10;

    // Total number of alive cells
    int total = 0;

    // Total size of grid in bytes
    size_t bytes = sizeof(int) * (dim + 2) * (dim + 2);

    // Allocate square grid of (dim+2)^2 elements, 2 added for ghost cells
    int *grid = (int *)malloc(bytes);

    // Allocate newGrid
    int *restrict newGrid = (int *)malloc(bytes);

    // Assign initial population randomly
    srand(SRAND_VALUE);
    for (i = 1; i <= dim; i++)
    {
        for (j = 1; j <= dim; j++)
        {
            grid[idx(i, j)] = rand() % 2;
        }
    }

    int fullSize = (dim + 2) * (dim + 2);
    gameTimer.start();
        for (iter = 0; iter < maxIter; iter++)
        {
            //writeGridToStdout(grid, iter, maxIter);
            // Left-Right columns
            for (i = 1; i <= dim; i++)
            {
                grid[idx(i, 0)] = grid[idx(i, dim)];     // Copy first real column to right ghost column
                grid[idx(i, dim + 1)] = grid[idx(i, 1)]; // Copy last real column to left ghost column
            }
            // Top-Bottom rows
            for (j = 0; j <= dim + 1; j++)
            {
                grid[idx(0, j)] = grid[idx(dim, j)];     // Copy first real row to bottom ghost row
                grid[idx(dim + 1, j)] = grid[idx(1, j)]; // Copy last real row to top ghost row
            }
            // Now we loop over all cells and determine their fate
            for (i = 1; i <= dim; i++)
            {
                for (j = 1; j <= dim; j++)
                {
                    // Get the number of neighbors for a given grid point
                    int numNeighbors = getNeighbors(grid, i, j);

                    // Here we have explicitly all of the game rules
                    if (grid[idx(i, j)] == 1 && numNeighbors < 2)
                        newGrid[idx(i, j)] = 0;
                    else if (grid[idx(i, j)] == 1 && (numNeighbors == 2 || numNeighbors == 3))
                        newGrid[idx(i, j)] = 1;
                    else if (grid[idx(i, j)] == 1 && numNeighbors > 3)
                        newGrid[idx(i, j)] = 0;
                    else if (grid[idx(i, j)] == 0 && numNeighbors == 3)
                        newGrid[idx(i, j)] = 1;
                    else
                        newGrid[idx(i, j)] = grid[idx(i, j)];
                }
            }

            // Can't switch pointers so we mannually have to copy array over
            for (i = 1; i <= dim; i++)
            {
                for (j = 1; j <= dim; j++)
                {
                    grid[idx(i, j)] = newGrid[idx(i, j)];
                }
            }
        }

        
            for (i = 1; i <= dim; i++)
            {
                for (j = 1; j <= dim; j++)
                {
                    total += grid[idx(i, j)];
                }
            }
        
    
    gameTimer.stop();

    printf("%d;%d;%lf\n",dim,total, gameTimer.getTime());
    fflush(stdout);

    // Release memory
    free(grid);
    free(newGrid);

    return 0;
}