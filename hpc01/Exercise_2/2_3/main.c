


#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
/* #include <unistd.h> */
#include <math.h>
#include <err.h>

double pkt_size_root(int h)
{
	return pow(2,h);
}

int main(int argc, char **argv)
{
	const int pkt_variate = 23;
	const int iterations = 1000;

	int ret = 0;

	char *inbuffer;
	char *outbuffer;

	double round_start;
	double round_end;
	int n_process;
	int rank;
	MPI_Status status;

	long int total_data_size = 0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_process);

	if (rank == 0) {
		printf("Rank of the communicator: %d\n", rank);
		printf("Number of processes: %d\n", n_process);
	}

	for (int h = 0; h <= pkt_variate; h++) {
		if (rank == 0) {
			outbuffer= (char*) malloc(pkt_size_root(h)*sizeof(char));
			if (outbuffer == NULL)
				errx(-2, "Failed to allocate output buffer!");
		} else if (rank == 1) {
			inbuffer = (char*) malloc(pkt_size_root(h)*sizeof(char));
			if (inbuffer == NULL)
				errx(-2, "Failed to allocate input buffer!");
		}

		for (int meas_repetitions = 0 ; meas_repetitions < 10; meas_repetitions++) {
			// Only rank 0 steers the communication
			if (rank == 0) {
				total_data_size = pkt_size_root(h)*iterations;

				// --------------------------------------------------------------------------
				// Blocking send
				// --------------------------------------------------------------------------
				round_start = MPI_Wtime();

				for (int i = 0; i < iterations; i++) {
					ret = MPI_Send(outbuffer,pkt_size_root(h),MPI_CHAR,1,0,MPI_COMM_WORLD);
					if (ret)
						errx(ret, "Failed to send on rank %d!", rank);
				}
				// --------------------------------------------------------------------------

				// --------------------------------------------------------------------------
				// Non-blocking send
				// --------------------------------------------------------------------------
				/* MPI_Request send_statuses[iterations]; */

				/* round_start = MPI_Wtime(); */

				/* for (int i = 0; i < iterations; i++) { */
				/* 	ret = MPI_Isend(outbuffer,pkt_size_root(h),MPI_CHAR,1,0,MPI_COMM_WORLD, &send_statuses[i]); */
				/* 	/\* printf("%d/%d: %.0f B message sent, ret: %d\n", rank, i, pow(1024,h), ret); *\/ */
				/* 	if (ret) */
				/* 		errx(ret, "Failed to send on rank %d!", rank); */
				/* } */

				/* MPI_Waitall(iterations, send_statuses, MPI_STATUSES_IGNORE); */
				// --------------------------------------------------------------------------
				round_end = MPI_Wtime();

				// Take end time and calculate
				double calc_throughput = (total_data_size/(round_end - round_start))/1000000;
				printf("Pkt size: %.0f -> Throughput %f MBps\n",
				pkt_size_root(h), calc_throughput);

			} else if (rank == 1) {
				for (int i = 0; i < iterations; i++) {
					ret = MPI_Recv(inbuffer,pkt_size_root(h),MPI_CHAR,0,0,MPI_COMM_WORLD, &status);
					if (ret)
						errx(ret, "Failed to receive on rank %d!", rank);
				}
			}
		}

		if (rank == 0)
			free(outbuffer);
		else if (rank == 1)
			free(inbuffer);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
