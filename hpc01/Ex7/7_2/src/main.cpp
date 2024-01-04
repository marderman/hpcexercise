#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <getopt.h>
#include <math.h>
#include <unistd.h>
#include <stdbool.h>
#include <vector>

// #define DEBUG

const static float TIMESTEP = 1e-3;   // s
const static float GAMMA = 6.673e-11; // (Nm^2)/(kg^2)
int NUM_BODIES, N_ITERATIONS, n_processes, rank, objects_per_process;


//Struct for 
struct float3
{
    float x, y, z;

    // Default Constructor
    float3() : x(0.0f), y(0.0f), z(0.0f) {}

    // Constructor with parameters to initialize members
    float3(float xVal, float yVal, float zVal)
        : x(xVal), y(yVal), z(zVal) {}
};

struct float4
{
    float x, y, z, w;

    // Default constructor with initializer list
    float4() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}

    // Constructor with parameters to initialize members
    float4(float xVal, float yVal, float zVal, float wVal)
        : x(xVal), y(yVal), z(zVal), w(wVal) {}
};
//Struct Definition for Bodies
typedef struct Body_t
{
    int id;
    float4 posMass; /* x = x */
                    /* y = y */
                    /* z = z */
                    /* w = Mass */
    float3 velocity;
    /* x = v_x */
    /* y = v_y */
    /* z = v_z */

    float3 acceleration;

    float3 force;

} Body_t;

// Define functions
//This are functions defined for Steps in the programm.
// 1. Initialize
// 2. Calculate
// 3. Update new Positions 
void initializeBodies();
void computeBodies();
void updatePosition(Body_t &body);

// Body Calculation Functions
// Functions used for the specific calculations needed for the new Position for the bodys
void calculateSpeed(float mass, float3 &currentSpeed, float3 force);
float getDistance(float4 a, float4 b);
void bodyBodyInteraction(Body_t &calculatedBody, Body_t interactingBody);


// Log Functions
void showAllData(Body_t *Body, int size, int rank);

// ROOT BODIES Variable for all Bodies
std::vector<Body_t> BODIES;

// Local Bodies Variable for slave Process
std::vector<Body_t> bodies;

MPI_Datatype MPI_FLOAT3, MPI_FLOAT4;
MPI_Datatype MPI_BODY_POS_VEL;

bool print_result = false;

int main(int argc, char *argv[])
{

    double start, stop;
    NUM_BODIES = 16;
    N_ITERATIONS = 200;

    int opt;
    while ((opt = getopt(argc, argv, "b:n:p")) != -1)
    {
        switch (opt)
        {
        case 'b':
            NUM_BODIES = std::atoi(optarg);
            break;
        case 'n':
            N_ITERATIONS = std::atoi(optarg);
            break;
        case 'p':
            print_result = true;
            break;
        default:
            std::cerr << "Usage: example -b <num_bodies> -n <num_iterations>" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

 
    MPI_Init(&argc, &argv);

    // Get the global rank and global size
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
{
        if (NUM_BODIES < n_processes)
    {
        printf("Starting with more processes than bodies!!!! Wrong configuration");
        MPI_Finalize();
        return -1;
    }
}
    #ifdef DEBUG
    printf("Running with %d processes\n", n_processes);
    #endif
    // Calculate the amount of rows each process has to compute (Set to minimum of 1)
    objects_per_process = std::max(NUM_BODIES / n_processes, 1);


    // Create an MPI type for float3
    MPI_Type_contiguous(3, MPI_FLOAT, &MPI_FLOAT3);
    MPI_Type_commit(&MPI_FLOAT3);

    // Create an MPI type for float4
    MPI_Type_contiguous(4, MPI_FLOAT, &MPI_FLOAT4);
    MPI_Type_commit(&MPI_FLOAT4);

    // Create MPI Datatype for the information that is distributed
    // It uses information stored in the struct so we need only one struct and update the information in the struct
    // So only updated informations has to be distributed.
    int blockLengths[3] = {1, 3, 3};
    MPI_Aint offsets[3];
    offsets[0] = offsetof(Body_t, id);
    offsets[1] = offsetof(Body_t, posMass);
    offsets[2] = offsetof(Body_t, velocity);

    MPI_Datatype types[3] = {MPI_INT, MPI_FLOAT4, MPI_FLOAT3};
    MPI_Type_create_struct(3, blockLengths, offsets, types, &MPI_BODY_POS_VEL);
    MPI_Type_commit(&MPI_BODY_POS_VEL);

    // Initialize with Data Distribution. Data is distributed throgh root process. With MPI_Bcast
    initializeBodies();

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        #ifdef DEBUG
        std::cout << "Start data" << std::endl;
        showAllData(BODIES.data(), NUM_BODIES, rank);
        std::cout << std::endl << "****Start CALCULATION****" << std::endl << std::endl;
        #endif
        start = MPI_Wtime();
    }

    // Calculate the Bodies
    // Each Process calculates only a subset of bodies. Determined with the amount of MPI Processes (objects_per_process)
    for (size_t i = 0; i < N_ITERATIONS; i++)
    {
        computeBodies();
    }


    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        stop = MPI_Wtime();
        if (print_result){
            std::cout << "Data after Simulation" << std::endl;
            showAllData(BODIES.data(), NUM_BODIES, rank);
        }
        double time_taken = (stop - start) / N_ITERATIONS;
        // std::cout << "The time for a n-body problem with " << NUM_BODIES << " bodies took " << time_taken << std::endl;
        std::cout << n_processes << "," << NUM_BODIES << "," << time_taken << std::endl;
    }
    
    MPI_Type_free(&MPI_BODY_POS_VEL);
    MPI_Type_free(&MPI_FLOAT3);
    MPI_Type_free(&MPI_FLOAT4);
    MPI_Finalize();
}

void communicate()
{
        //Use MPI_Allgather to gather information from each process Each process sends only their specific supset 
        //
        MPI_Allgather(BODIES.data() + (rank*objects_per_process), objects_per_process, MPI_BODY_POS_VEL,
                BODIES.data(),objects_per_process,MPI_BODY_POS_VEL,MPI_COMM_WORLD);
        // Received Data
        MPI_Barrier(MPI_COMM_WORLD);
        #ifdef DEBUG
        if(rank == 0){
            std::cout << "Data recievied" << std::endl;
            showAllData(BODIES.data(),NUM_BODIES,0);
        }
        #endif
}

void initializeBodies()
{
    BODIES.resize(NUM_BODIES);
    if (rank == 0)
    {

        srand(0); // Always the same random numbers
        for (int i = 0; i < NUM_BODIES; i++)
        {
            BODIES[i].id = i;
            BODIES[i].posMass.x = 1e-8 * static_cast<float>(rand()); // Modify the random values to
            BODIES[i].posMass.y = 1e-8 * static_cast<float>(rand()); // increase the position changes
            BODIES[i].posMass.z = 1e-8 * static_cast<float>(rand()); // and the velocity
            BODIES[i].posMass.w = 1e4 * static_cast<float>(rand());
            BODIES[i].velocity.x = 0.0f;
            BODIES[i].velocity.y = 0.0f;
            BODIES[i].velocity.z = 0.0f;
            BODIES[i].acceleration.x = 0.0f;
            BODIES[i].acceleration.y = 0.0f;
            BODIES[i].acceleration.z = 0.0f;
            BODIES[i].force.x = 0.0f;
            BODIES[i].force.y = 0.0f;
            BODIES[i].force.z = 0.0f;
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    int err = MPI_Bcast(BODIES.data(), NUM_BODIES, MPI_BODY_POS_VEL, 0, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
    {
        char error_string[100];
        int length_of_error_string;
        MPI_Error_string(err, error_string, &length_of_error_string);
        std::cerr << error_string << std::endl;
        // Exit or take corrective action
    }
}

void computeBodies()
{
    for (int i = rank*objects_per_process; i < ((rank+1)*objects_per_process); i++)
    {
        BODIES[i].force.x = 0;
        BODIES[i].force.y = 0;
        BODIES[i].force.z = 0;
        BODIES[i].acceleration.x = 0;
        BODIES[i].acceleration.y = 0;
        BODIES[i].acceleration.z = 0;
        for (int j = 0; j < NUM_BODIES; j++)
        {
            if (i != j)
            {
                bodyBodyInteraction(BODIES[i], BODIES[j]);
            }
        }
        calculateSpeed(BODIES[i].posMass.w, BODIES[i].velocity, BODIES[i].force);
        updatePosition(BODIES[i]);
    }
    #ifdef DEBUG
    if (rank == 0){
        std::cout << "After calculation" << std::endl;
        showAllData(BODIES.data(),NUM_BODIES,0);

    }
    #endif
    communicate();
}

void updatePosition(Body_t &body)
{
    body.posMass.x += body.velocity.x * TIMESTEP;
    body.posMass.y += body.velocity.y * TIMESTEP;
    body.posMass.z += body.velocity.z * TIMESTEP;
}

void calculateSpeed(float mass, float3 &currentSpeed, float3 force)
{

    currentSpeed.x += (force.x / mass) * TIMESTEP;
    currentSpeed.y += (force.y / mass) * TIMESTEP;
    currentSpeed.z += (force.z / mass) * TIMESTEP;
}

void bodyBodyInteraction(Body_t &calculatedBody, Body_t interactingBody)
{
    float distance = getDistance(calculatedBody.posMass, interactingBody.posMass);

    if (distance == 0)
        return;
    calculatedBody.force.x += -GAMMA * ((calculatedBody.posMass.w * interactingBody.posMass.w) / pow(distance, 2)) * (interactingBody.posMass.x - calculatedBody.posMass.x);
    calculatedBody.force.y += -GAMMA * ((calculatedBody.posMass.w * interactingBody.posMass.w) / pow(distance, 2)) * (interactingBody.posMass.y - calculatedBody.posMass.y);
    calculatedBody.force.z += -GAMMA * ((calculatedBody.posMass.w * interactingBody.posMass.w) / pow(distance, 2)) * (interactingBody.posMass.z - calculatedBody.posMass.z);

    // calculateSpeed(calculatedBody->posMass.w, calculatedBody.velocity, calculatedBody.force);
}

float getDistance(float4 a, float4 b)
{
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;

    return sqrtf(dx * dx + dy * dy + dz * dz);
    // Calculate distance of two particles
}

void showAllData(Body_t *Body, int size, int rank)
{
    int a;
    for (a = 0; a < size; a++)
    {
        // printf("id %d pos_x %.4f pos_y %.4f pos_z %.4f v_x %.1f v_y %.1f v_z %.1f a_x %.1f a_y %.1f a_z %.1f m %.1f F_x %.1f F_y %.1f F_z %.1f\n",
        //        Body[a].id, Body[a].posMass.x, Body[a].posMass.y, Body[a].posMass.z,
        //        Body[a].velocity.x, Body[a].velocity.y, Body[a].velocity.z,
        //        Body[a].acceleration.x, Body[a].acceleration.y, Body[a].acceleration.z,
        //        Body[a].posMass.w,
        //        Body[a].force.x, Body[a].force.y, Body[a].force.z);
        std::cout << "id " << Body[a].id
            << std::setprecision(2)
            << ", pos_x " << Body[a].posMass.x
            << ", pos_y " << Body[a].posMass.y
            << ", pos_z " << Body[a].posMass.z
            << ", v_x " << Body[a].velocity.x
            << ", v_y " << Body[a].velocity.y
            << ", v_z " << Body[a].velocity.z
            << ", a_x " << Body[a].acceleration.x
            << ", a_y " << Body[a].acceleration.y
            << ", a_z " << Body[a].acceleration.z
            << ", m " << Body[a].posMass.w
            << ", F_x " << Body[a].force.x
            << ", F_y " << Body[a].force.y
            << ", F_z " << Body[a].force.z
            << std::endl;

    }
    std::cout << std::endl << std::endl;
}
