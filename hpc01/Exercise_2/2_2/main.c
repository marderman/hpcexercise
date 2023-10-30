#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#define iterations 1000

int main(int argc, char **argv){

    int n_process,rank,start=1;
    double round_start, round_end;
    char* inbuffer;
    char* outbuffer;
    double* time_measurements_round;
    time_measurements_round = (double*)malloc(iterations*sizeof(double));
    MPI_Status status;
    outbuffer = "Ping-Pong";

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_process);
    /*if(rank == 0){
        printf("N-Tasks: %d\n", n_process);
    }
    MPI_Barrier(MPI_COMM_WORLD); */

    for (size_t h = 0; h < 3; h++){
    inbuffer = (char*) malloc(1*pow(1024,h)*sizeof(char));
    outbuffer= (char*) malloc(1*pow(1024,h)*sizeof(char));
    for (size_t i = 0; i < iterations; i++){
    if (rank == 0 && start == 1)
        {
        round_start = MPI_Wtime();
        MPI_Send(outbuffer,1*pow(1024,h)*sizeof(char),MPI_CHAR,(rank + 1) % n_process,0,MPI_COMM_WORLD );
        start = 0;
        }
    else 
            {
            MPI_Recv(inbuffer,1*pow(1024,h)*sizeof(char),MPI_CHAR,MPI_ANY_SOURCE,0,MPI_COMM_WORLD, &status);
            if (rank == 0)
                {
                round_end = MPI_Wtime();
                time_measurements_round[i] = (double) round_end - round_start;
                //printf("Test %f\n", round_end);
                round_start = MPI_Wtime();
                }
        //printf("Hello from Process %d\n", rank); printf("Nachricht %s von Prozess %d\n",inbuffer, rank); fflush(stdout);
            outbuffer = inbuffer;
            MPI_Send(outbuffer,1*pow(1024,h)*sizeof(char),MPI_CHAR,(rank + 1) % n_process,0,MPI_COMM_WORLD );
            }
        }
    if (rank == 0) { /* use time on master node */
        double average_round, average_half_round;
        for (size_t i = 0; i < iterations; i++)
            {
            average_round += time_measurements_round[i];
            }
        average_round /= iterations;
        average_half_round = average_round/2;
        printf("Average full roundtrip time %lf for message size of %f bytes\n", average_round*1000,pow(1024,h));
        printf("Average half roundtrip time %lf for message size of %f bytes\n", average_half_round*1000,pow(1024,h));
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    free(inbuffer);
    free(outbuffer);
    MPI_Finalize();

    return 0;
   
}