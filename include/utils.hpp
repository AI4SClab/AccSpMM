#pragma once
#include <cuda_runtime.h>
#include <string>
#include <cstring>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file,
           const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s: %d\n", cudaGetErrorString(err), file, line);
        fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, const int line) {
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s at %s: %d\n", cudaGetErrorString(err), file, line);
        fprintf(stderr, "%s\n", cudaGetErrorString(err));
    }
}

void printCudaInfo() {
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024.0 * 1024.0));
    // printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    printf("   WarpSize:   %d\n", deviceProps.warpSize);
    printf("   shMem/Blk:  %ld\n", deviceProps.sharedMemPerBlock);
    printf("   L2Size:     %.0f MB\n", static_cast<float>(deviceProps.l2CacheSize) / (1024.0 * 1024.0));
  }
  printf("---------------------------------------------------------\n");
}

std::string match_filename(std::string s) {
    int last_slash      = s.rfind('/') + 1;
    std::string suffix  = s.substr(last_slash, s.size()-last_slash);
    return suffix;
}

void init_vec(const vint rows, const vint cols, MAT_VAL_TYPE** Mat) {
    for(int i = 0; i < rows; ++i) {
        for(int j = 0; j < cols; ++j) {
            Mat[i][j] = 1.0;
        }
    }
}

void init_vec1(const vint nnz, MAT_VAL_TYPE* Mat, MAT_VAL_TYPE val) {
    for(vint i = 0; i < nnz; ++i) {
        Mat[i] = val;
    }
}

void init_vecB(const vint rows, const vint cols, MAT_VAL_TYPE* Mat, MAT_VAL_TYPE val) {
    vint t = 1;
    for(vint i = 0; i < rows; ++i) {
        for(vint j = 0; j < cols; ++j) {
            Mat[i * cols + j] = t;
        }
        t += 1;
    }
}

bool balanceStrategy(METCFBit<MAT_VAL_TYPE>& metcf_bit, vint numTcBlocks, vint numBlocks) {
    vint AvgTCBlock = numTcBlocks / numBlocks;
    vint sumWindow = 0;
    for(vint i = 1; i < metcf_bit.rowWindowOffset.size() - 1; ++i) {
        vint current_tcblocks = metcf_bit.rowWindowOffset[i] - metcf_bit.rowWindowOffset[i-1];
        sumWindow += std::abs(int(current_tcblocks - AvgTCBlock));
    }
    vint IBD = sumWindow / numBlocks;

    return IBD > IMBALANCE_THRESHOLD;
}

struct GpuTimer {
    cudaEvent_t start;
    cudaEvent_t stop;
    GpuTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start() { cudaEventRecord(start); }
    void Stop()  { cudaEventRecord(stop); }

    float Elapsed() {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

