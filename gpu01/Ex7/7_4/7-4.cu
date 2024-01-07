#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for compute intensity calculation on an m x m grid
__global__ void computeIntensityKernel(int m, float* result) {
    // Get the thread indices
    int i = threadIdx.x;
    int j = threadIdx.y;

    // Dimensions of the grid
    int n = m;

    // Calculate compute intensity
    *result = (2.0 * n * n * m * m) / (n * n + (n - m + 1) * (n - m + 1));
}

// Function to calculate compute intensity on GPU for an m x m grid
float computeIntensityGPU(int m) {
    // Allocate memory for result on the GPU
    float *d_result;
    cudaMalloc((void**)&d_result, sizeof(float));

    // Define the grid dimensions
    dim3 blockDim(m, m);

    // Launch the kernel with the specified grid dimensions
    computeIntensityKernel<<<1, blockDim>>>(m, d_result);

    // Copy the result from GPU to CPU
    float h_result;
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_result);

    return h_result;
}

int main() {
    // Values of m for which we want to calculate compute intensity
    int m_values[] = {3, 5, 7, 11};
    int num_m_values = sizeof(m_values) / sizeof(m_values[0]);

    // Calculate and print compute intensity for each m value on GPU
    printf("\nCompute Intensity for different m values:\n");
    for (int i = 0; i < num_m_values; ++i) {
        float ci = computeIntensityGPU(m_values[i]);
        printf("m = %d: %.2f\n", m_values[i], ci);
    }

    return 0;
}
