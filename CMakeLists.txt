cmake_minimum_required(VERSION 3.18)
project(AccSpMM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 80 89 90)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})

add_compile_options(-O3 -fopenmp-simd)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/mma_tf32.cu)

add_executable(mma ${SOURCES} ${CUDA_SOURCES})

find_package(CUDAToolkit REQUIRED)
target_link_libraries(mma CUDA::cudart)
