#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cstdlib>
#include <chTimer.hpp>

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <MB>" << std::endl;
        return 1;
    }

    ChTimer memCpyH2DTimer, memCpyD2HTimer;
	ChTimer kernelTimer;

    int mb = std::atoi(argv[1]);

    std::cout << "Reducing: " << mb << "MB elements\n"; 
    // Size of the array
    int size = mb*1024*1024;
    std::cout << size;

    // Create a device_vector on the GPU
    
    thrust::device_vector<int> d_vec(size);
    // Initialize the device_vector with the value 1 using thrust::fill
    thrust::fill(d_vec.begin(), d_vec.end(), 1);
    // Use thrust::reduce to calculate the sum
    kernelTimer.start();
    int sum = thrust::reduce(d_vec.begin(), d_vec.end());
   

    // Print the result
    std::cout << "Sum: " << sum << std::endl;
    kernelTimer.stop();

    std::cout << kernelTimer.getTime();

    return 0;
}