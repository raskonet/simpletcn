cmake_minimum_required(VERSION 3.18)
project(TCN_HYBRID LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
# Enable OpenMP for the CPU Backward Pass
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -fopenmp")

# Check for CUDA (Optional)
find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    message(STATUS "✅ CUDA Found. Enabling GPU Acceleration.")
    add_definitions(-DUSE_CUDA)
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    set(CUDA_SOURCES src/cuda_ops.cu)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
else()
    message(STATUS "⚠️  CUDA Not Found. Compiling in CPU-Only Mode.")
    set(CUDA_SOURCES "")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
find_package(OpenMP REQUIRED)

file(GLOB CPP_SOURCES "src/*.cpp")

add_executable(tcn_app ${CPP_SOURCES} ${CUDA_SOURCES})

target_link_libraries(tcn_app PRIVATE OpenMP::OpenMP_CXX)
if(CUDAToolkit_FOUND)
    target_link_libraries(tcn_app PRIVATE CUDA::cudart)
endif()

# Also build the tests to ensure no regression
file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(testfile ${TEST_SOURCES})
    get_filename_component(testname ${testfile} NAME_WE)
    add_executable(${testname} ${testfile} ${CPP_SOURCES} ${CUDA_SOURCES})
    # Exclude main.cpp from tests to avoid duplicate main()
    set_target_properties(${testname} PROPERTIES SOURCES "${testfile};src/tensor.cpp;src/conv1d.cpp;src/residual_block.cpp;src/tcn.cpp;src/relu.cpp;src/dropout.cpp;src/utils.cpp;${CUDA_SOURCES}")
    target_link_libraries(${testname} PRIVATE OpenMP::OpenMP_CXX)
    if(CUDAToolkit_FOUND)
        target_link_libraries(${testname} PRIVATE CUDA::cudart)
    endif()
endforeach()
