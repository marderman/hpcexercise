#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define iterations 20000


int main(int argc, char** argv)
{

    double total_start, total_end, round_start,round_end;
    int start = 1;

    double* time_measurements;
    time_measurements = (double*)malloc(iterations*sizeof(double));

    char* inbuffer;
    char* outbuffer;

    inbuffer = (char*) malloc(1*1024*1024*sizeof(char));
    outbuffer= (char*) malloc(1*1024*1024*sizeof(char));


    MPI_Status status;
    outbuffer = "Hello dies ist eine Testnachricht";

    MPI_Init(&argc, &argv);

    int n_process;
    MPI_Comm_size(MPI_COMM_WORLD, &n_process);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        printf("Zeit: %lf ;Anzahl Prozesse %d\n",MPI_Wtime(), n_process);
    }

    char hostname[256];
    gethostname(hostname, 256);
    for (size_t i = 0; i < n_process; i++)
    {
        if (rank == i){
       printf("Zeit: %lf; Prozess %d von %d läuft auf %s\n", MPI_Wtime(),rank+1,n_process,hostname);
        }
    }
    


    MPI_Barrier(MPI_COMM_WORLD); 
    if (rank == 0) {
    printf("Zeit: %lf; START\n", MPI_Wtime());
    printf("Zeit: %lf; Tasks;Runtime;Average Message Roundtrip Time;Average Message Send Time\n",MPI_Wtime());
    fflush(stdout);
}
    MPI_Barrier(MPI_COMM_WORLD);
    total_start = MPI_Wtime();


    for (size_t i = 0; i < iterations; i++)
    {
    
    if (rank == 0 && start ==1)
    {
        //Start Time Measurement
        round_start = MPI_Wtime();
        MPI_Send(outbuffer,100*sizeof(char),MPI_CHAR,(rank + 1) % n_process,0,MPI_COMM_WORLD );
        start = 0;
    }

    else 
    {

        MPI_Recv(inbuffer,100*sizeof(char),MPI_CHAR,MPI_ANY_SOURCE,0,MPI_COMM_WORLD, &status);
        if (rank == 0)
        {
            round_end = MPI_Wtime();
            time_measurements[i] = (double) round_end- round_start;
           
        }
        //printf("Hello from Process %d\n", rank); printf("Nachricht %s von Prozess %d\n",inbuffer, rank); fflush(stdout);
        outbuffer = inbuffer;
        if (rank == 0)
        {
             round_start = MPI_Wtime();
        }
        MPI_Send(outbuffer,100*sizeof(char),MPI_CHAR,(rank + 1) % n_process,0,MPI_COMM_WORLD );
    }
     /* code */
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    total_end = MPI_Wtime();

    MPI_Finalize();

    if (rank == 0) { /* use time on master node */
        //printf("Runtime = %f\n", total_end-total_start);
        double average;
        for (size_t i = 0; i < iterations; i++)
        {
            /* code */
        average += time_measurements[i];
        }
        average /= iterations;
        //printf("Average Message Rounttrip Tiem %lf\n", average);
        double average_per_message = average / ((double) n_process);
        //printf("Average Message Send Time %lf",average_per_message);
        printf("Zeit: %lf;%d;%f;%f;%f\n",MPI_Wtime(),n_process,total_end - total_start,average,average_per_message);
    }
    
}