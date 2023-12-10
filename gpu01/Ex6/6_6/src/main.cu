#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

int main() {
    // Size of the array
    const int size = 4096;

    // Create a device_vector on the GPU
    thrust::device_vector<int> d_vec(size);

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