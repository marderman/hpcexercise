/******************************************************************************
 *
 *           XXXII Heidelberg Physics Graduate Days - GPU Computing
 *
 *                 Gruppe : TODO
 *
 *                   File : main.cu
 *
 *                Purpose : n-Body Computation
 *
 ******************************************************************************/

#include <cmath>
#include <ctime>
#include <iostream>
#include <cstdlib>
#include <chCommandLine.h>
#include <chTimer.hpp>
#include <cstdio>
#include <iomanip>

const static int DEFAULT_NUM_ELEMENTS = 1024;
const static int DEFAULT_NUM_ITERATIONS = 5;
const static int DEFAULT_BLOCK_DIM = 128;

const static float TIMESTEP = 1e-3;	  // s
const static float GAMMA = 6.673e-11; // (Nm^2)/(kg^2)

//
// Structures
//
// Here with two AOS (arrays of structures).
//
struct Body_t
{
	float4 *posMass;  /* x = x */
					  /* y = y */
					  /* z = z */
					  /* w = Mass */
	float3 *velocity; /* x = v_x */
					  /* y = v_y */
					  /* z = v_z */

	Body_t() : posMass(NULL), velocity(NULL) {}
};

struct Body_t_soa
{
	float* x;
	float* y;
	float* z;
	float* w;

	float* vx;
	float* vy;
	float* vz;

	Body_t_soa() : x(NULL), y(NULL), z(NULL),w(NULL), vx(NULL),vy(NULL), vz(NULL) {}
};
//
// Function Prototypes
//
void printHelp(char *);
void printElement(Body_t, int, int);

//
// Device Functions
//

//
// Calculate the Distance of two points
//
__device__ float
getDistance(float4 a, float4 b)
{
	float dx = a.x - b.x;
	float dy = a.y - b.y;
	float dz = a.z - b.z;

	return sqrtf(dx * dx + dy * dy + dz * dz);
	// Calculate distance of two particles
}

//
// Calculate the forces between two bodies
//
__device__ void
bodyBodyInteraction(float4 bodyA, float4 bodyB, float3 &force)
{
	float distance = getDistance(bodyA, bodyB);

	if (distance == 0)
		return;

	force.x += -GAMMA * ((bodyA.w * bodyB.w) / pow(distance, 2)) * (bodyB.x - bodyA.x);
	force.y += -GAMMA * ((bodyA.w * bodyB.w) / pow(distance, 2)) * (bodyB.y - bodyA.y);
	force.z += -GAMMA * ((bodyA.w * bodyB.w) / pow(distance, 2)) * (bodyB.z - bodyA.z);
}

//
// Calculate the new velocity of one particle
//
__device__ void
calculateSpeed(float mass, float3 &currentSpeed, float3 force)
{
	currentSpeed.x += (force.x / mass) * TIMESTEP;
	currentSpeed.y += (force.y / mass) * TIMESTEP;
	currentSpeed.z += (force.z / mass) * TIMESTEP;
}

//
// n-Body Kernel for the speed calculation
//
__global__ void
simpleNbody_Kernel(int numElements, float4 *bodyPos, float3 *bodySpeed)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	float4 elementPosMass;
	float3 elementForce;
	float3 elementSpeed;

	if (elementId < numElements)
	{
		elementPosMass = bodyPos[elementId];
		elementSpeed = bodySpeed[elementId];
		elementForce = make_float3(0, 0, 0);

		for (int i = 0; i < numElements; i++)
		{
			if (i != elementId)
			{
				bodyBodyInteraction(elementPosMass, bodyPos[i], elementForce);
			}
		}

		calculateSpeed(elementPosMass.w, elementSpeed, elementForce);

		bodySpeed[elementId] = elementSpeed;
	}
}

__global__ void
sharedNbody_Kernel(int numElements, Body_t_soa* dBody)
{
	// Use the packed values and SOA to optimize load and store operations
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	__shared__ float x[877];
	__shared__ float y[877];
	__shared__ float z[877];
	__shared__ float w[877];
	__shared__ float vx[877];
	__shared__ float vz[877];
	__shared__ float vy[877];

	//Berechnete Werte
	float elementPosX,elementPosY,elementPosZ;
	float elementAccelX, elementAccelY, elementAccellZ;

	float3 elementForce = make_float3(0,0,0);

	if (elementId < numElements)
	{
		printf("sizeof(dBody) %d ", sizeof(dBody->x[elementId]));

		printf("dBody->x[elementId] %f", dBody->x[elementId]);
		printf("dBody->y[elementId] %f", dBody->y[elementId]);
		printf("dBody->z[elementId] %f", dBody->z[elementId]);
		printf("dBody->w[elementId] %f", dBody->w[elementId]);
		float4 bodyB = make_float4(dBody->x[elementId], dBody->y[elementId],dBody->z[elementId], dBody->w[elementId]);
		printf("bodyB");
		float3 elementSpeed = make_float3(dBody->vx[elementId], dBody->vy[elementId], dBody->vz[elementId]);
		printf("elementSpeed");
		

		for (size_t i = 0; i < gridDim.x; i++)
		{
			/* code */
			printf("x[threadIdx.x]");
			x[threadIdx.x] = dBody->x[threadIdx.x +  i*blockDim.x];
			y[threadIdx.x] = dBody->y[threadIdx.x +  i*blockDim.x];
			z[threadIdx.x] = dBody->z[threadIdx.x +  i*blockDim.x];
			vx[threadIdx.x] = dBody->vx[threadIdx.x + i*blockDim.x];
			vy[threadIdx.x] = dBody->vy[threadIdx.x + i*blockDim.x];
			vz[threadIdx.x] = dBody->vz[threadIdx.x + i*blockDim.x];
			__syncthreads();
			printf("write value");

			for (size_t i = 0; i < 877; i++)
			{	
				float4 bodyA = make_float4(x[threadIdx.x], y[threadIdx.x], z[threadIdx.x], w[threadIdx.x]);
				bodyBodyInteraction(bodyA, bodyB, elementForce);
			}

			calculateSpeed(bodyB.w, elementSpeed,elementForce);
			dBody->x[elementId] = bodyB.x;
			dBody->y[elementId] = bodyB.y;
			dBody->z[elementId] = bodyB.z;
			dBody->vx[elementId] = elementSpeed.x + vx[threadIdx.x];
			dBody->vy[elementId] = elementSpeed.y + vy[threadIdx.y];
			dBody->vz[elementId] = elementSpeed.z + vz[threadIdx.z];
			__syncthreads();

			
		}
/* code */
	}

}






//
// n-Body Kernel to update the position
// Needed to prevent write-after-read-hazards
//
__global__ void
updatePosition_Kernel(int numElements, float4 *bodyPos, float3 *bodySpeed)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	float4 elementPosMass;
	float3 elementSpeed;

	if (elementId < numElements)
	{
		elementPosMass = bodyPos[elementId];
		elementSpeed = bodySpeed[elementId];
		elementPosMass.x += elementSpeed.x * TIMESTEP;
		elementPosMass.y += elementSpeed.y * TIMESTEP;
		elementPosMass.z += elementSpeed.z * TIMESTEP;
		bodyPos[elementId] = elementPosMass;
	}
}

__global__ void
SoAUpdatePosition_Kernel(int numElements, Body_t_soa * SoA)
{
	int elementId = blockIdx.x * blockDim.x + threadIdx.x;

	if (elementId < numElements)
	{
		SoA->x[elementId] += SoA->vx[elementId] * TIMESTEP;
		SoA->y[elementId] += SoA->vy[elementId] * TIMESTEP;
		SoA->z[elementId] += SoA->vz[elementId] * TIMESTEP;
	}
}


void allocateAOS(bool pinnedMemory, Body_t &h_particles, int numElements);

void allocateSOA(bool pinnedMemory, Body_t_soa &h_particles,int numElements);

void initializeAOS(int numElements, Body_t &h_particles);

void initializeSOA(int numElements, Body_t_soa &h_particles);

void allocateDeviceMemoryAOS(Body_t &d_particles, int numElements, Body_t &h_particles);

void allocateDeviceMemorySOA(Body_t_soa &d_particles, int numElements, Body_t_soa &h_particles);


//
// Main
//
int main(int argc, char *argv[])
{
int sizeShMem = 49152;
	bool showHelp = chCommandLineGetBool("h", argc, argv);
	if (!showHelp)
	{
		showHelp = chCommandLineGetBool("help", argc, argv);
	}

	if (showHelp)
	{
		printHelp(argv[0]);
		exit(0);
	}

	std::cout << "***" << std::endl
			  << "*** Starting ..." << std::endl
			  << "***" << std::endl;

	ChTimer memCpyH2DTimer, memCpyD2HTimer;
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
	bool pinnedMemory = chCommandLineGetBool("p", argc, argv);
	if (!pinnedMemory)
	{
		pinnedMemory = chCommandLineGetBool("pinned-memory", argc, argv);
	}
	Body_t h_particles;
	Body_t_soa h_particles_soa;
	Body_t d_particles;
	Body_t_soa d_particles_soa;

    bool memoryLayout = chCommandLineGetBool("soa" , argc, argv);
	if (memoryLayout)
	{
		     	allocateSOA(pinnedMemory, h_particles_soa, numElements);
				initializeSOA(numElements, h_particles_soa);
				allocateDeviceMemorySOA(d_particles_soa, numElements, h_particles_soa);
    }
	if (!memoryLayout)
	{
		printf("*** Using AOS");
		allocateAOS(pinnedMemory, h_particles, numElements);
		initializeAOS(numElements, h_particles);
    	allocateDeviceMemoryAOS(d_particles, numElements, h_particles);
		std::cout << "aos ready allocated mem" << std::endl;
       
    }


    //
	// Copy Data to the Device
	//
	memCpyH2DTimer.start();

	if (memoryLayout)
	{
		cudaMemcpy(d_particles_soa.x, h_particles_soa.x,
				static_cast<size_t>(numElements * sizeof(float)),cudaMemcpyHostToDevice);
		cudaMemcpy(d_particles_soa.y, h_particles_soa.y,
				static_cast<size_t>(numElements * sizeof(float)),cudaMemcpyHostToDevice);
		cudaMemcpy(d_particles_soa.z, h_particles_soa.z,
				static_cast<size_t>(numElements * sizeof(float)),cudaMemcpyHostToDevice);
		cudaMemcpy(d_particles_soa.vx, h_particles_soa.vx,
				static_cast<size_t>(numElements * sizeof(float)),cudaMemcpyHostToDevice);
		cudaMemcpy(d_particles_soa.vy, h_particles_soa.vy,
				static_cast<size_t>(numElements * sizeof(float)),cudaMemcpyHostToDevice);
		cudaMemcpy(d_particles_soa.vz, h_particles_soa.vz,
				static_cast<size_t>(numElements * sizeof(float)),cudaMemcpyHostToDevice);
	}
	else {
		cudaMemcpy(d_particles.posMass, h_particles.posMass,
			   static_cast<size_t>(numElements * sizeof(float4)),
			   cudaMemcpyHostToDevice);
		cudaMemcpy(d_particles.velocity, h_particles.velocity,
			   static_cast<size_t>(numElements * sizeof(float3)),
			   cudaMemcpyHostToDevice);
	}

	memCpyH2DTimer.stop();

	//
	// Get Kernel Launch Parameters
	//
	int blockSize = 0,
		gridSize = 0,
		numIterations = 0;

	// Number of Iterations
	chCommandLineGet<int>(&numIterations, "i", argc, argv);
	chCommandLineGet<int>(&numIterations, "num-iterations", argc, argv);
	numIterations = numIterations != 0 ? numIterations : DEFAULT_NUM_ITERATIONS;

	// Block Dimension / Threads per Block
	chCommandLineGet<int>(&blockSize, "t", argc, argv);
	chCommandLineGet<int>(&blockSize, "threads-per-block", argc, argv);
	blockSize = blockSize != 0 ? blockSize : DEFAULT_BLOCK_DIM;

	if (blockSize > 1024)
	{
		std::cout << "\033[31m***" << std::endl
				  << "*** Error - The number of threads per block is too big" << std::endl
				  << "***\033[0m" << std::endl;

		exit(-1);
	}

	gridSize = ceil(static_cast<float>(numElements) / static_cast<float>(blockSize));

	dim3 grid_dim = dim3(gridSize);
	dim3 block_dim = dim3(blockSize);

	std::cout << "***" << std::endl;
	std::cout << "*** Grid: " << gridSize << std::endl;
	std::cout << "*** Block: " << blockSize << std::endl;
	std::cout << "***" << std::endl;

	bool silent = chCommandLineGetBool("silent", argc, argv);

	kernelTimer.start();

	for (int i = 0; i < numIterations; i++)
	{
		if(memoryLayout)
		{
			sharedNbody_Kernel<<<grid_dim, block_dim, sizeShMem>>>(numElements, &d_particles_soa);
			SoAUpdatePosition_Kernel<<<grid_dim, block_dim>>>(numElements, &d_particles_soa);

			// cudaMemcpy(h_particles.posMass, d_particles.posMass, sizeof(float4), cudaMemcpyDeviceToHost);
			// cudaMemcpy(h_particles.velocity, d_particles.velocity, sizeof(float3), cudaMemcpyDeviceToHost);
			// if (!silent)
			// {
			// 	printElement(h_particles, 0, i + 1);
			// }
		}
		else{
			simpleNbody_Kernel<<<grid_dim, block_dim>>>(numElements, d_particles.posMass,
													d_particles.velocity);
			updatePosition_Kernel<<<grid_dim, block_dim>>>(numElements, d_particles.posMass,
														d_particles.velocity);

			cudaMemcpy(h_particles.posMass, d_particles.posMass, sizeof(float4), cudaMemcpyDeviceToHost);
			cudaMemcpy(h_particles.velocity, d_particles.velocity, sizeof(float3), cudaMemcpyDeviceToHost);
			if (!silent)
			{
				printElement(h_particles, 0, i + 1);
			}
		}
		
	}

	// Synchronize
	cudaDeviceSynchronize();

	// Check for Errors
	cudaError_t cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess)
	{
		std::cout << "\033[31m***" << std::endl
				  << "***ERROR*** " << cudaError << " - " << cudaGetErrorString(cudaError)
				  << std::endl
				  << "***\033[0m" << std::endl;

		return -1;
	}

	kernelTimer.stop();

	//
	// Copy Back Data
	//
	memCpyD2HTimer.start();

	cudaMemcpy(h_particles.posMass, d_particles.posMass,
			   static_cast<size_t>(numElements * sizeof(*(h_particles.posMass))),
			   cudaMemcpyDeviceToHost);
	cudaMemcpy(h_particles.velocity, d_particles.velocity,
			   static_cast<size_t>(numElements * sizeof(*(h_particles.velocity))),
			   cudaMemcpyDeviceToHost);

	memCpyD2HTimer.stop();

	if (memoryLayout)
	{
	// Free Memory
	if (!pinnedMemory)
	{
		free(h_particles.posMass);
		free(h_particles.velocity);
	}
	else
	{
		cudaFreeHost(h_particles.posMass);
		cudaFreeHost(h_particles.velocity);
	}
	}

	cudaFree(d_particles.posMass);
	cudaFree(d_particles.velocity);

	// Print Meassurement Results
	std::cout << "***" << std::endl
			  << "*** Results:" << std::endl
			  << "***    Num Elements: " << numElements << std::endl
			  << "***    Num Iterations: " << numIterations << std::endl
			  << "***    Threads per block: " << blockSize << std::endl
			  << "***    Time to Copy to Device: " << 1e3 * memCpyH2DTimer.getTime()
			  << " ms" << std::endl
			  << "***    Copy Bandwidth: "
			  << 1e-9 * memCpyH2DTimer.getBandwidth(numElements * sizeof(h_particles))
			  << " GB/s" << std::endl
			  << "***    Time to Copy from Device: " << 1e3 * memCpyD2HTimer.getTime()
			  << " ms" << std::endl
			  << "***    Copy Bandwidth: "
			  << 1e-9 * memCpyD2HTimer.getBandwidth(numElements * sizeof(h_particles))
			  << " GB/s" << std::endl
			  << "***    Time for n-Body Computation: " << 1e3 * kernelTimer.getTime()
			  << " ms" << std::endl
			  << "***" << std::endl;

	return 0;
}

void allocateDeviceMemoryAOS(Body_t &d_particles, int numElements, Body_t &h_particles)
{
	printf("*** Allocating Device Memory\n");
    cudaMalloc(&(d_particles.posMass),static_cast<size_t>(numElements * sizeof(*(d_particles.posMass))));
    cudaMalloc(&(d_particles.velocity),static_cast<size_t>(numElements * sizeof(*(d_particles.velocity))));
	
	//printf("h_particles.posMass %d, d_particles.posMass %d\n h_particles.velocity %d, d_particles.velocity", h_particles.posMass, d_particles.posMass, h_particles.velocity, d_particles.velocity);

    if (h_particles.posMass == NULL || h_particles.velocity == NULL ||
        d_particles.posMass == NULL || d_particles.velocity == NULL)
    {
        std::cout << "\033[31m***" << std::endl
                  << "*** Error - Memory allocation failed" << std::endl
                  << "***\033[0m" << std::endl;

        exit(-1);
    }
}

void allocateDeviceMemorySOA(Body_t_soa &d_particles, int numElements, Body_t_soa &h_particles)
{
	printf("*** Allocating Device Memory\n");
	cudaMalloc(&(d_particles.x),
		static_cast<size_t>(numElements*sizeof(*(d_particles.x))));
	cudaMalloc(&(d_particles.y),
		static_cast<size_t>(numElements*sizeof(*(d_particles.y))));
	cudaMalloc(&(d_particles.z),
		static_cast<size_t>(numElements*sizeof(*(d_particles.z))));
	cudaMalloc(&(d_particles.w),
		static_cast<size_t>(numElements*sizeof(*(d_particles.w))));
	cudaMalloc(&(d_particles.vx),
		static_cast<size_t>(numElements*sizeof(*(d_particles.vx))));
	cudaMalloc(&(d_particles.vy),
		static_cast<size_t>(numElements*sizeof(*(d_particles.vy))));
	cudaMalloc(&(d_particles.vz),
		static_cast<size_t>(numElements*sizeof(*(d_particles.vz))));

	if (h_particles.x == NULL || h_particles.y == NULL ||
        d_particles.x == NULL || d_particles.y == NULL ||
		
		h_particles.z == NULL || h_particles.w == NULL ||
        d_particles.z == NULL || d_particles.w == NULL ||

		h_particles.vx == NULL || h_particles.vy == NULL ||
        d_particles.vx == NULL || d_particles.vy == NULL ||

		h_particles.vz == NULL ||
        d_particles.vz == NULL
		)
    {
        std::cout << "\033[31m***" << std::endl
                  << "*** Error - Memory allocation failed" << std::endl
                  << "***\033[0m" << std::endl;

        exit(-1);
    }
}

void initializeAOS(int numElements, Body_t &h_particles)
{
	printf("*** Initialize Data");

    // Init Particles
    //	srand(static_cast<unsigned>(time(0)));
    srand(0); // Always the same random numbers
    for (int i = 0; i < numElements; i++)
    {
        h_particles.posMass[i].x = 1e-8 * static_cast<float>(rand()); // Modify the random values to
        h_particles.posMass[i].y = 1e-8 * static_cast<float>(rand()); // increase the position changes
        h_particles.posMass[i].z = 1e-8 * static_cast<float>(rand()); // and the velocity
        h_particles.posMass[i].w = 1e4 * static_cast<float>(rand());
        h_particles.velocity[i].x = 0.0f;
        h_particles.velocity[i].y = 0.0f;
        h_particles.velocity[i].z = 0.0f;
    }

    printElement(h_particles, 0, 0);
}

void initializeSOA(int numElements, Body_t_soa &h_particles)
{
	printf("*** Initialize Data");
	srand(0);
	for (int i = 0;i < numElements;i++)
	{
		h_particles.x[i] = 1e-8 * static_cast<float>(rand()); // Modify the random values to
		h_particles.y[i] = 1e-8 * static_cast<float>(rand()); // increase the position changes
        h_particles.z[i] = 1e-8 * static_cast<float>(rand()); // and the velocity
        h_particles.w[i] = 1e4 * static_cast<float>(rand());
        h_particles.vx[i] = 0.0f;
        h_particles.vy[i] = 0.0f;
        h_particles.vz[i] = 0.0f;
	}
}

void allocateSOA(bool pinnedMemory, Body_t_soa &h_particles, int numElements)
{
	printf("*** Allocate Host Memory");
    if (!pinnedMemory)
    {
		h_particles.x = static_cast<float *>(malloc(static_cast<size_t>(numElements* sizeof(*(h_particles.x)))));
		h_particles.y = static_cast<float *>(malloc(static_cast<size_t>(numElements*sizeof(*(h_particles.y)))));
		h_particles.z = static_cast<float *>(malloc(static_cast<size_t>(numElements*sizeof(*(h_particles.z)))));
		h_particles.w = static_cast<float *>(malloc(static_cast<size_t>(numElements*sizeof(*(h_particles.w)))));
		h_particles.vx = static_cast<float*>(malloc(static_cast<size_t>(numElements*sizeof(*(h_particles.vx)))));
		h_particles.vy = static_cast<float*>(malloc(static_cast<size_t>(numElements*sizeof(*(h_particles.vy)))));
		h_particles.vz = static_cast<float*>(malloc(static_cast<size_t>(numElements*sizeof(*(h_particles.vz)))));
    }
}

void allocateAOS(bool pinnedMemory, Body_t &h_particles, int numElements)
{
	printf("*** Allocate Host Memory");
    if(!pinnedMemory)
    {
        // Pageable
        h_particles.posMass = static_cast<float4 *>
		(malloc(static_cast<size_t>(numElements * sizeof(*(h_particles.posMass)))));
        h_particles.velocity = static_cast<float3 *>
		(malloc(static_cast<size_t>(numElements * sizeof(*(h_particles.velocity)))));
    }
    else
    {
        // Pinned
        cudaMallocHost(&(h_particles.posMass),
                       static_cast<size_t>(numElements * sizeof(*(h_particles.posMass))));
        cudaMallocHost(&(h_particles.velocity),
                       static_cast<size_t>(numElements * sizeof(*(h_particles.velocity))));
    }
}

void printHelp(char *argv)
{
	std::cout << "Help:" << std::endl
			  << "  Usage: " << std::endl
			  << "  " << argv << " [-p] [-s <num-elements>] [-t <threads_per_block>]"
			  << std::endl
			  << "" << std::endl
			  << "  -p|--pinned-memory" << std::endl
			  << "    Use pinned Memory instead of pageable memory" << std::endl
  			  << "" << std::endl
			  << "  --soa" << std::endl
			  << "    Use SOA Allocation instead of AOS Allocation" << std::endl
			  << "" << std::endl
			  << "  -s <num-elements>|--size <num-elements>" << std::endl
			  << "    Number of elements (particles)" << std::endl
			  << "" << std::endl
			  << "  -i <num-iterations>|--num-iterations <num-iterations>" << std::endl
			  << "    Number of iterations" << std::endl
			  << "" << std::endl
			  << "  -t <threads_per_block>|--threads-per-block <threads_per_block>"
			  << std::endl
			  << "    The number of threads per block" << std::endl
			  << "" << std::endl
			  << "  --silent"
			  << std::endl
			  << "    Suppress print output during iterations (useful for benchmarking)" << std::endl
			  << "" << std::endl;
}

//
// Print one element
//
void printElement(Body_t particles, int elementId, int iteration)
{
	float4 posMass = particles.posMass[elementId];
	float3 velocity = particles.velocity[elementId];

	std::cout << "***" << std::endl
			  << "*** Printing Element " << elementId << " in iteration " << iteration << std::endl
			  << "***" << std::endl
			  << "*** Position: <"
			  << std::setw(11) << std::setprecision(9) << posMass.x << "|"
			  << std::setw(11) << std::setprecision(9) << posMass.y << "|"
			  << std::setw(11) << std::setprecision(9) << posMass.z << "> [m]" << std::endl
			  << "*** velocity: <"
			  << std::setw(11) << std::setprecision(9) << velocity.x << "|"
			  << std::setw(11) << std::setprecision(9) << velocity.y << "|"
			  << std::setw(11) << std::setprecision(9) << velocity.z << "> [m/s]" << std::endl
			  << "*** Mass: "
			  << std::setw(11) << std::setprecision(9) << posMass.w << " kg" << std::endl
			  << "***" << std::endl;
}
