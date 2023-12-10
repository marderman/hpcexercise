#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cstdlib>

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <MB>" << std::endl;
        return 1;
    }

    int mb = std::atoi(argv[1]);

    std::cout << "Reducing: " << mb << "MB elements\n"; 
    // Size of the array
    int size = mb*1024*1024;
    std::cout << size;

    // Create a device_vector on the GPU
    thrust::device_vector<float> d_vec(size);

    // Initialize the device_vector with some values
    for (int i = 0; i < size; ++i) {
        d_vec[i] = 1;
    }

    // Use thrust::reduce to calculate the sum
    int sum = thrust::reduce(d_vec.begin(), d_vec.end());

    // Print the result
    std::cout << "Sum: " << sum << std::endl;

    return 0;
}