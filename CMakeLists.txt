cmake_minimum_required(VERSION 3.18)
project(TCN_CUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Optimization Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fopenmp")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(OpenMP REQUIRED)

# Find all sources
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

add_executable(tcn_app ${SOURCES})

target_link_libraries(tcn_app PRIVATE OpenMP::OpenMP_CXX)

# Copy data to build folder
file(COPY ${CMAKE_SOURCE_DIR}/data/ DESTINATION ${CMAKE_BINARY_DIR})
