#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <ctype.h>


void my_barrier(MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int tag = 0;
    MPI_Status status;

    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Recv(NULL, 0, MPI_INT, i, tag, comm, &status);
        }
    } else {
        MPI_Send(NULL, 0, MPI_INT, 0, tag, comm);
    }
}

int main(int argc, char *argv[]) {
    
    int iterations;
    iterations =  atoi(argv[1]);

    /*if(isdigit(atoi(argv[1]))){
        iterations =  atoi(argv[1]);
    }
    else{
        return 1;
    }*/

    double round_start, round_stop;

    double* time_measurements_own; 
    double* time_measurements_mpi;
    time_measurements_own = (double*)malloc(iterations*sizeof(double));
    time_measurements_mpi = (double*)malloc(iterations*sizeof(double));

    MPI_Init(&argc, &argv);
    
    int rank, size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    for(int i=0; i<iterations; i++){
        
        if(rank==0){
            round_start = MPI_Wtime();
        }
        
        my_barrier(MPI_COMM_WORLD);
        
        if(rank==0){
            round_stop = MPI_Wtime();
        }
        
        time_measurements_own[i] = (double) round_stop - round_start;
    }

    for(int i=0; i<iterations; i++){
    
        if(rank==0){
            round_start = MPI_Wtime();
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        
        if(rank==0){
            round_stop = MPI_Wtime();
        }
        
        time_measurements_mpi[i] = (double) round_stop - round_start;
    
    }

    if (rank == 0) { /* use time on master node */
            double average_round, average_mpi_round;

            for (size_t i = 0; i < iterations; i++) {
                average_round += time_measurements_own[i];
                average_mpi_round += time_measurements_mpi[i];
            }

            average_round /= iterations;
            average_mpi_round /= iterations;
            printf("Average time when using own implementation of barrier %lf for %d iterations\n", average_round*1e6,iterations);
            printf("Average time when using MPI_BARRIER %lf for %d iterations\n", average_mpi_round*1e6,iterations);
    }
    
    free(time_measurements_mpi);
    free(time_measurements_own);

    MPI_Finalize();

    return 0;
}
