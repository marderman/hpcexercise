cmake_minimum_required(VERSION 3.27)

# find_package(EnvModules REQUIRED)
# env_module(purge)
# env_module(load devtoolset/10)
# env_module(load cuda/12.0)
# env_module_list(my_modules_found)
# message(modules=${my_modules_found})

project(Exercise4_2 LANGUAGES CXX CUDA) 

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
#target_compile_features(Exercise4_2 PUBLIC cxx_std_11)

# We need to explicitly state that we need all CUDA files in the 
# particle library to be built with -dc as the member functions 
# could be called by other libraries and executables
#set_target_properties( Exercise4_2
 #                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(CMAKE_CUDA_ARCHITECTURES 75)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/inc ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # Add your include directory here

add_executable(programm src/main.cpp)

target_sources(programm PRIVATE
    src/main.cpp
    src/kernel.cu
)

set_source_files_properties(src/kernel.cu PROPERTIES LANGUAGE CUDA)

target_include_directories(programm PRIVATE ${CMAKE_SOURCE_DIR}/inc /opt/cuda-11.4/include)