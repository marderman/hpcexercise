#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define DELTA_T 5
#define NBODY_DEFAULT 4 
#define TIME_DEFAULT 0.001
// #define GAMMA -6.673*1e-11
#define GAMMA -0.000001
#define RAND_TH(a) RAND_MAX * a
// #define DEBUG

typedef struct {
    double mass;
    double pos_x;
    double pos_y;
    double pos_z;
    double v_x;
    double v_y;
    double v_z;
    double a_x;
    double a_y;
    double a_z;
    double F_x;
    double F_y;
    double F_z;
    int id;
}BodyProp;

typedef struct {
    double mass;
    double pos_x;
    double pos_y;
    double pos_z;
    int id;
}BodyPropRot;

void showData(BodyProp* Bodys, int size);
void showDataLog(char* string, BodyProp* Bodys, int size, int rank);
void showAllData(BodyProp* Body, int size, int rank);

int main(int argc ,char * argv[]) {
    int number_of_processes, rank, nBody, timeIter, bodysPerThread, i, j, time, r;
    MPI_Status status;
    MPI_Request request;
    BodyProp *Body = NULL;
    BodyPropRot *BodyRot = NULL;
    BodyProp *subBody = NULL;
    BodyPropRot *rotBody = NULL;
    struct timeval start, end;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &number_of_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int resciever =  ((rank + 1) % number_of_processes);
    
    // if(rank == 0)
    // {
    //     printf("Start");fflush(stdout);
    // }

    /* create a type for struct BodyProp */
    const int nitems=14;
    int blocklengths[14] = {    1,1,1,
                                1,1,1,
                                1,1,1,
                                1,1,1,
                                1,1};
    MPI_Datatype types[14] = {  MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
                                MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
                                MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
                                MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
                                MPI_DOUBLE, MPI_INT};
    MPI_Datatype mpi_BodyProp;
    MPI_Aint     offsets[14];

    offsets[0] = offsetof(BodyProp, mass);
    offsets[1] = offsetof(BodyProp, pos_x);
    offsets[2] = offsetof(BodyProp, pos_y);
    offsets[3] = offsetof(BodyProp, pos_z);
    offsets[4] = offsetof(BodyProp, v_x);
    offsets[5] = offsetof(BodyProp, v_y);
    offsets[6] = offsetof(BodyProp, v_z);
    offsets[7] = offsetof(BodyProp, a_x);
    offsets[8] = offsetof(BodyProp, a_y);
    offsets[9] = offsetof(BodyProp, a_z);
    offsets[10] = offsetof(BodyProp, F_x);
    offsets[11] = offsetof(BodyProp, F_y);
    offsets[12] = offsetof(BodyProp, F_z);
    offsets[13] = offsetof(BodyProp, id);

    const int nitemsRot=5;
    int blocklengthsRot[5] = {  1,1,1,
                                1,1};
    MPI_Datatype typesRot[5] = {   MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, 
                                    MPI_DOUBLE, MPI_INT};
    MPI_Datatype mpi_BodyPropRot;
    MPI_Aint     offsetsRot[5];

    offsetsRot[0] = offsetof(BodyPropRot, mass);
    offsetsRot[1] = offsetof(BodyPropRot, pos_x);
    offsetsRot[2] = offsetof(BodyPropRot, pos_y);
    offsetsRot[3] = offsetof(BodyPropRot, pos_z);
    offsetsRot[4] = offsetof(BodyPropRot, id);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_BodyProp);
    MPI_Type_commit(&mpi_BodyProp);

    MPI_Type_create_struct(nitemsRot, blocklengthsRot, offsetsRot, typesRot, &mpi_BodyPropRot);
    MPI_Type_commit(&mpi_BodyPropRot);

    if(rank == 0)
    {
        // get args
        if (argc > 1) 
        {
            nBody = atoi(argv[1]);  // Parse size from command line argument
            timeIter = atoi(argv[2]);
        } 
        else 
        {
            nBody = NBODY_DEFAULT;
            timeIter = TIME_DEFAULT;
        }
        
        Body = (BodyProp*)malloc(nBody * sizeof(BodyProp));
        BodyRot = (BodyPropRot*)malloc(nBody * sizeof(BodyPropRot));

        for(i = 1; i < number_of_processes; i++)
        {
            MPI_Send(&nBody, sizeof(nBody)/sizeof(int), MPI_INT, i, 1, MPI_COMM_WORLD);
            MPI_Send(&timeIter, sizeof(timeIter)/sizeof(int), MPI_INT, i, 2, MPI_COMM_WORLD);
        }

        // initialisation 
        for(i = 0; i < nBody; i++)
        {
            Body[i].mass = (double)rand() / RAND_TH(250000000);
            Body[i].pos_x = (double)rand() / RAND_TH(100);
            Body[i].pos_y = (double)rand() / RAND_TH(100);
            Body[i].pos_z = (double)rand() / RAND_TH(100);
            BodyRot[i].mass = Body[i].mass;
            BodyRot[i].pos_x = Body[i].pos_x;
            BodyRot[i].pos_y = Body[i].pos_y;
            BodyRot[i].pos_z = Body[i].pos_z;
            BodyRot[i].id = i;
            // Body[i-1].mass = 250000000*i;
            // Body[i-1].pos_x = 100*i;
            // Body[i-1].pos_y = 100*i;
            // Body[i-1].pos_z = 100*i;
            Body[i].a_x = 0;
            Body[i].a_y = 0;
            Body[i].a_z = 0;
            Body[i].v_x = 0;
            Body[i].v_y = 0;
            Body[i].v_z = 0;
            Body[i].F_x = 0;
            Body[i].F_y = 0;
            Body[i].F_z = 0;
            Body[i].id = i;
#ifdef DEBUG
            showAllData(Body, nBody, rank);
#endif
        }
        // showData(Body, nBody);
        // printf("time: %d ring: %d ", time, r);
        // showDataLog("init", Body, nBody, rank);
    }
    else
    {
        MPI_Recv(&nBody, sizeof(nBody)/sizeof(int), MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&timeIter, sizeof(timeIter)/sizeof(int), MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);
    }

    bodysPerThread = nBody / number_of_processes;
    if(bodysPerThread = 0)
    {
        printf("more processes than bodys");fflush(stdout);
        return -1;
    }

    // printf("bodyPerThread: %d, nBody: %d, size2Transmit: %d\n", bodysPerThread, nBody, (bodysPerThread * sizeof(Body))/(sizeof(double)*10));
    subBody = (BodyProp*)malloc(bodysPerThread * sizeof(BodyProp));
    rotBody = (BodyPropRot*)malloc(bodysPerThread * sizeof(BodyPropRot));
    
    //Scatter data
    MPI_Scatter(Body, bodysPerThread, mpi_BodyProp, subBody, bodysPerThread, mpi_BodyProp, 0, MPI_COMM_WORLD);
    MPI_Scatter(BodyRot, bodysPerThread, mpi_BodyPropRot, rotBody, bodysPerThread, mpi_BodyPropRot, 0, MPI_COMM_WORLD);

#ifdef DEBUG
    showAllData(subBody, bodysPerThread, rank);
#endif
    if(rank == 0)
    {
        // printf("scattered Data");fflush(stdout);
        gettimeofday(&start, NULL);
    }
    for(time = 0; time < timeIter; time++)
    {
        // showData(subBody, bodysPerThread);
        // printf("time: %d ring: %d ", time, r);
        // showDataLog("mcpy", rotBody, bodysPerThread, rank);

        for(r = 0; r < number_of_processes; r++)
        {
            // compute internal forces
            for(j = 0; j < bodysPerThread; j++)
            {
                subBody[j].F_x = 0;
                subBody[j].F_y = 0;
                subBody[j].F_z = 0;

                for(i = 0; i < bodysPerThread; i++)
                {
                    if(subBody[j].id == rotBody[i].id)
                    {
                        // printf("equal\n");
                        continue;
                    }
                    else{
                        // printf("unequal\n");
                    }
                    double rx = subBody[j].pos_x - rotBody[i].pos_x;
                    double ry = subBody[j].pos_y - rotBody[i].pos_y;
                    double rz = subBody[j].pos_z - rotBody[i].pos_z;
                    double rdist_sq = rx * rx + ry * ry + rz * rz;
                    double rdist = sqrt(rdist_sq);
                    rx = rx / rdist;
                    ry = ry / rdist;
                    rz = rz / rdist;
                    // printf("dist %lf\n",rdist);
                    // printf("rank %d Body i: %d Body j: %d r: %lf\n", rank, i, j, r);
                    subBody[j].F_x += ((GAMMA * rotBody[i].mass * subBody[j].mass * rx) / rdist_sq);
                    subBody[j].F_y += ((GAMMA * rotBody[i].mass * subBody[j].mass * ry) / rdist_sq);
                    subBody[j].F_z += ((GAMMA * rotBody[i].mass * subBody[j].mass * rz) / rdist_sq);
                    // printf("sub id %d\n", subBody[j].id);
                    // showAllData(subBody, bodysPerThread, rank);
                    // printf("rot id %d\n", rotBody[j].id);
                    // showAllData(rotBody, bodysPerThread, rank);
                }
                
            }   
            // printf("time: %d ring: %d ", time, r);
            // showDataLog("rotBody before", rotBody, bodysPerThread, rank);
            // printf("time: %d ring: %d ", time, r);
            // showDataLog("subBody before", subBody, bodysPerThread, rank);

            MPI_Send(rotBody, bodysPerThread, mpi_BodyPropRot, resciever, 1, MPI_COMM_WORLD);
            MPI_Recv(rotBody, bodysPerThread, mpi_BodyPropRot, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);

            // printf("time: %d ring: %d ", time, r);
            // showDataLog("rotBody after", rotBody, bodysPerThread, rank);
            // printf("time: %d ring: %d ", time, r);
            // showDataLog("subBody after", subBody, bodysPerThread, rank);
        }

        for (j = 0; j < bodysPerThread; j++)
        {
            subBody[j].a_x = subBody[j].F_x / subBody[j].mass;
            subBody[j].a_y = subBody[j].F_y / subBody[j].mass;
            subBody[j].a_z = subBody[j].F_z / subBody[j].mass;
            subBody[j].v_x = subBody[j].a_x * DELTA_T;
            subBody[j].v_y = subBody[j].a_y * DELTA_T;
            subBody[j].v_z = subBody[j].a_z * DELTA_T;
            subBody[j].pos_x += subBody[j].v_x * DELTA_T;
            subBody[j].pos_y += subBody[j].v_y * DELTA_T;
            subBody[j].pos_z += subBody[j].v_z * DELTA_T;
            // printf("rank: %d id: %d, Fx: %.1lf Fy: %.1lf Fz: %.1lf\n", rank, subBody[j].id, subBody[j].F_x, subBody[j].F_y, subBody[j].F_z);
        }
        // if(time % 10 == 0)
        MPI_Barrier(MPI_COMM_WORLD);
        if(0)
        {
            MPI_Gather(subBody, bodysPerThread, mpi_BodyProp, Body, bodysPerThread, mpi_BodyProp, 0, MPI_COMM_WORLD);
            if(rank == 0)
            {
                showData(Body, nBody);
            }
        }
        // if(rank == 0)
        // {
        //     printf("%d", time);fflush(stdout);
        // }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(subBody, bodysPerThread, mpi_BodyProp, Body, bodysPerThread, mpi_BodyProp, 0, MPI_COMM_WORLD);
    if(rank == 0)
    {
        gettimeofday(&end, NULL);
        double elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1e6;
        printf("Nbody,%d,numberofProcesses,%d,time,%lfus\n", nBody,number_of_processes,elapsed_time);fflush(stdout);
        // showData(Body, nBody);
    }

    MPI_Finalize();

    free(subBody);
    free(rotBody);
    free(Body);
}

void showData(BodyProp* Bodys, int size)
{
    int a;
    for(a = 0; a < size; a++)
    {
        printf("%d;%.1lf;%.1lf;%.1lf;", Bodys[a].id, Bodys[a].pos_x, Bodys[a].pos_y, Bodys[a].pos_z);
    }
    printf("\n");
}

void showDataLog(char* string, BodyProp* Bodys, int size, int rank)
{
    printf("rank: %d %s \n", rank, string);
    int a;
    for(a = 0; a < size; a++)
    {
        // printf("id;%d;pos_x;%.1lf;pos_y;%.1lf;pos_z;%.1lf\n", Bodys[a].id, Bodys[a].pos_x, Bodys[a].pos_y, Bodys[a].pos_z);

        printf("id;%d;pos_x;%.1f;pos_y;%.1f;pos_z;%.1f;v_x;%.1f;v_y;%.1f;v_z;%.1f;a_x;%.1f;a_y;%.1f;a_z;%.1f;m;%.1f;F_x;%.1f;F_y;%.1f;F_z;%.1f\n", 
        Bodys[a].id, Bodys[a].pos_x, Bodys[a].pos_y,Bodys[a].pos_z,
        Bodys[a].v_x,Bodys[a].v_y,Bodys[a].v_z,
        Bodys[a].a_x,Bodys[a].a_y,Bodys[a].a_z,
        Bodys[a].mass, 
        Bodys[a].F_x, Bodys[a].F_y, Bodys[a].F_z);
    }
    printf("\n\n");
}

void showAllData(BodyProp* Body, int size, int rank)
{
    int a;
    for(a = 0; a < size; a++)
    {
        printf("id %d pos_x %.1f pos_y %.1f pos_z %.1f v_x %.1f v_y %.1f v_z %.1f a_x %.1f a_y %.1f a_z %.1f m %.1f F_x %.1f F_y %.1f F_z %.1f\n", 
        Body[a].id, Body[a].pos_x, Body[a].pos_y,Body[a].pos_z,
        Body[a].v_x,Body[a].v_y,Body[a].v_z,
        Body[a].a_x,Body[a].a_y,Body[a].a_z,
        Body[a].mass, 
        Body[a].F_x, Body[a].F_y, Body[a].F_z);
    }
    printf("\n\n");
}