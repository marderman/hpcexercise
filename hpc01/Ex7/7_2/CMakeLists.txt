cmake_minimum_required(VERSION 3.27)

find_package(EnvModules REQUIRED)
env_module(purge)
env_module(load devtoolset/10)
env_module(load mpi/open-mpi-4.1.6)
env_module_list(my_modules_found)
message(modules=${my_modules_found})

project(Exercise7_2 VERSION 1.0 LANGUAGES CXX)

# Enable MPI
find_package(EnvModules REQUIRED)
env_module(purge)
env_module(load devtoolset/10)
env_module(load mpi/open-mpi-4.1.6)
env_module_list(my_modules_found)
message(modules=${my_modules_found})
find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
#target_compile_features(Exercise4_2 PUBLIC cxx_std_11)

# Add include directories (old version legacy)
# include_directories(${CMAKE_SOURCE_DIR}/inc) # Add your include directory here

add_executable(programm src/main.cpp)

# Add the -O3 optimization flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

target_sources(programm PRIVATE
    src/main.cpp
    #src/grid.cpp
)

# Find Math Library
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(programm PUBLIC ${MATH_LIBRARY})
endif()

target_include_directories(programm PRIVATE ${CMAKE_SOURCE_DIR}/inc)

link_directories(${CMAKE_SOURCE_DIR}/lib)
target_link_libraries(programm PUBLIC MPI::MPI_CXX)
