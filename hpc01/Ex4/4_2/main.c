#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <err.h>
#include <math.h>
#include <ncurses.h>

// The positional parameters are (keep the order):
//
// 1. size of the matrix (must be grater than or equal 4)
// 2. number of iterations to measure on (greater than 0)
// 3. (optional) switch "-p" to print the evolution of the matrix
//      - slows the execution significantly


double compute_element(int row, int column, int grid_size, double** g)
{
	double ret = -69.0;
	double interm_sum = 0;

	if (row != 0)
		interm_sum += g[row-1][column];
	if (row != grid_size-1)
		interm_sum += g[row+1][column];
	if (column != 0)
		interm_sum += g[row][column-1];
	if (column != grid_size-1)
		interm_sum += g[row][column+1];

	ret = g[row][column] + 0.24 * ((-4.0)* g[row][column] + interm_sum);

	return ret;
}

void iterate_matrix(double** a, double** b, int size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size ; j++) {
			b[i][j] = compute_element(i, j, size, a);
		}
	}
}

void print_matrix(double** g, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			// Calculate the logarithm (base 2) of the matrix element
			double log_value = log2(g[i][j]);
			int color_pair = 0;

			// Set text color based on the logarithmic value
			if (log_value < 0.5) {
				color_pair = 1;
			} else if (log_value < 1.0) {
				color_pair = 2;
			} else if (log_value < 1.5) {
				color_pair = 3;
			} else if (log_value < 2.0) {
				color_pair = 4;
			} else if (log_value < 2.5) {
				color_pair = 5;
			} else if (log_value < 3.0) {
				color_pair = 6;
			} else if (log_value < 3.5) {
				color_pair = 7;
			} else if (log_value < 4.0) {
				color_pair = 8;
			} else if (log_value < 4.5) {
				color_pair = 9;
			} else if (log_value < 5.0) {
				color_pair = 10;
			} else if (log_value < 5.5) {
				color_pair = 11;
			} else if (log_value < 6.0) {
				color_pair = 12;
			} else if (log_value < 6.5) {
				color_pair = 13;
			} else if (log_value < 7.0) {
				color_pair = 14;
			} else if (log_value < 7.5) {
				color_pair = 15;
			}

			attron(COLOR_PAIR(color_pair));
			addch(ACS_BLOCK);
			attroff(COLOR_PAIR(color_pair));
		}
		printw("\n");
	}
	refresh();
}

int main(int argc, char *argv[])
{
	int grid_size;
	int iterations;
	double** a_grid;
	double** b_grid;
	int i, j;
	struct timespec begin, end;

	// ---------------------------------------------------------------------------
	// Parsing input parameters
	// ---------------------------------------------------------------------------
	int req_print = 0;

	if (argc < 3)
		errx(-1,"Wrong number of arguments, at least two are expected!");

	grid_size = atoi(argv[1]);

	// Check if the conversion was successful
	if (grid_size < 4)
		errx(-1, "Too small grid, must be grater than 1!");

	iterations = atoi(argv[2]);

	if (iterations < 1)
		errx(-1, "Bad number of iterations for computations, must be more than 0!");

	if (argc == 4)
		if (strcmp(argv[3], "-p") == 0)
			req_print = 1;

	printf("Grid size set to %d, iterations: %d\n", grid_size, iterations);

	// ---------------------------------------------------------------------------
	// Allocate grids
	// ---------------------------------------------------------------------------
	a_grid = (double**)malloc(grid_size * sizeof(double*));
	b_grid = (double**)malloc(grid_size * sizeof(double*));
	if (a_grid == NULL || b_grid == NULL)
		errx(-1, "Array allocation failed!");

	for (i = 0; i < grid_size; i++) {
		a_grid[i] = (double*)malloc(grid_size * sizeof(double));
		b_grid[i] = (double*)malloc(grid_size * sizeof(double));
		if (a_grid[i] == NULL || b_grid[i] == NULL)
			errx(-1, "Array allocation failed!");
	}

	for (i = 0; i < grid_size; i++) {
		for (j = 0; j < grid_size; j++) {
			if (i == 0 && (double)j >= (grid_size/4.0) && (double)j <= (grid_size*3.0)/4.0)
				a_grid[i][j] = 127.0;
			else
				a_grid[i][j] = 0.0;

			b_grid[i][j] = 0.0;
		}
	}

	// ---------------------------------------------------------------------------
	// If printing was requested, then initialize the ncurses library
	// ---------------------------------------------------------------------------
	if (req_print) {
		initscr();
		start_color();
		init_pair(1, COLOR_BLUE, COLOR_BLACK);
		init_pair(2, COLOR_CYAN, COLOR_BLACK);
		init_pair(3, COLOR_GREEN, COLOR_BLACK);
		init_pair(4, COLOR_GREEN, COLOR_BLACK); // Light Green
		init_pair(5, COLOR_YELLOW, COLOR_BLACK);
		init_pair(6, COLOR_RED, COLOR_BLACK);
		init_pair(7, COLOR_MAGENTA, COLOR_BLACK);
		init_pair(8, COLOR_MAGENTA, COLOR_BLACK); // Magenta
		init_pair(9, COLOR_BLUE, COLOR_BLACK); // Light Blue
		init_pair(10, COLOR_CYAN, COLOR_BLACK); // Light Cyan
		init_pair(11, COLOR_YELLOW, COLOR_BLACK); // Brown/Yellow
		init_pair(12, COLOR_WHITE, COLOR_BLACK);
		init_pair(13, COLOR_WHITE, COLOR_BLACK); // Light Gray
		init_pair(14, COLOR_BLACK, COLOR_BLACK); // Dark Gray
		init_pair(15, COLOR_RED, COLOR_BLACK); // Light Red

		printw("Initial matrix\n");
		print_matrix(a_grid, grid_size);
	}

	// ---------------------------------------------------------------------------
	// Main part of the code
	// ---------------------------------------------------------------------------
	// Internal variable to switch matrices
	double** inter;

	clock_gettime(CLOCK_REALTIME, &begin);
	for (int k = 0; k < iterations; k++) {
		// Update matrix
		iterate_matrix(a_grid, b_grid, grid_size);

		// Print matrix on ncurses screen
		if (req_print) {
			clear();
			printw("Iteration %d\n", k);
			print_matrix(b_grid, grid_size);
			usleep(200000);
		}

		// Switch matrices, the result matrix b_grid is assigned to a_grid and the
		// previously output matrix a_grid is used to be overwritten by new values.
		inter = a_grid;
		a_grid = b_grid;
		b_grid = inter;
	}
	clock_gettime(CLOCK_REALTIME, &end);

	// Turn off the ncurses screen
	if(req_print) {
		endwin();
	}

	// ---------------------------------------------------------------------------
	// Result processing
	// ---------------------------------------------------------------------------
	// This is a bit complicated so let me elaborate (three additions to ops):
	// 1. The corners of the matrix have two operations less
	// 2. The edges without the corners have one operation less
	// 3. The rest of the matrix has all operations, thus 8
	//
	//         1             2                       3
	int ops = 4*6 + 7*4*(grid_size-2) + 8*(grid_size-2)*(grid_size-2);

	// Calculating the time elapsed
	long elapsed_s = end.tv_sec - begin.tv_sec;
	long elapsed_ns = end.tv_nsec - begin.tv_nsec;
	double double_elapsed = (double)(elapsed_s + elapsed_ns*1e-9);

	// Calculating the amount of FLOPS
	double avg_iter_duration = double_elapsed/(double)iterations;
	double flops = (ops/avg_iter_duration)*1e-9;

	printf("Ops: %d,  %f s, %f s/it, %f GFLOPS/s\n", ops, double_elapsed, avg_iter_duration, flops);

	// ---------------------------------------------------------------------------
	// Clean up
	// ---------------------------------------------------------------------------
	for (i = 0; i < grid_size; i++) {
		free(a_grid[i]);
		free(b_grid[i]);
	}
	free(a_grid);
	free(b_grid);
	return 0;
}
