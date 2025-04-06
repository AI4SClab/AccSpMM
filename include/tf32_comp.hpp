/**
 * @file tf32_comp.hpp
 * @author Haisha Zhao
 * @date 2025-04-02
 * 
 * @copyright MIT License (c) 2025 Haisha Zhao
*/

#pragma once
#include "common.hpp"
#include "mma_tf32.cuh"

class TF32Compute {
private:
    const int WARMUP_TIME;
    const int EXE_TIME;
    const int THRESHOLD;
    
    int dtmWarpsPerBlock(vint feature_dim) const {
        if(feature_dim <= THRESHOLD) {
            return feature_dim / (COL_WINDOW_R << 1);
        } else {
            return feature_dim / (COL_WINDOW_R << 2);
        }
    }
    
    void getKernelConfig(vint numBlocks, vint feature_dim, dim3& grid_size, dim3& block_size) const {
        int warpsPerBlk = dtmWarpsPerBlock(feature_dim);
        grid_size = dim3(numBlocks, 1, 1);
        block_size = dim3(WARP_SIZE, warpsPerBlk, 1);
    }
    
    template<typename KernelFunc, typename... Args>
    void executeKernel(GpuTimer& timer, KernelFunc kernel, dim3 grid_size, dim3 block_size, Args... args) const {
        for(int i = 0; i < WARMUP_TIME; ++i) {
            kernel<<<grid_size, block_size>>>(args...);
        }
        cudaDeviceSynchronize();
        
        timer.Start();
        for(int i = 0; i < EXE_TIME; ++i) {
            kernel<<<grid_size, block_size>>>(args...);
        }
        timer.Stop();
        cudaDeviceSynchronize();
    }
    
public:
    TF32Compute(vint warmup_time = 10, vint exe_time = 10, vint threshold = 512) 
        : WARMUP_TIME(warmup_time), EXE_TIME(exe_time), THRESHOLD(threshold) {}
    
    float compute(
        TCLOCAL_TYPE* d_tcLocalBit,
        MAT_PTR_TYPE* d_sparseA2B,
        MAT_VAL_TYPE* d_dataA, 
        MAT_VAL_TYPE* d_DenseMatB, 
        MAT_VAL_TYPE* d_DenseMatC, 
        vint* d_rowWindowOffset, 
        vint* d_metcOffset, 
        vint numNodes,
        vint numBlocks,
        vint feature_dim,
        GpuTimer& timer
    ) {
        dim3 grid_size, block_size;
        getKernelConfig(numBlocks, feature_dim, grid_size, block_size);
        
        std::cout << "doing pipelining mma ..." << std::endl;
        
        if (feature_dim <= THRESHOLD) {
            executeKernel(timer, tf32_mma_kernel, grid_size, block_size, 
                          d_tcLocalBit, d_sparseA2B, d_dataA, d_rowWindowOffset, 
                          d_metcOffset, d_DenseMatB, d_DenseMatC, numNodes, feature_dim);
        } else {
            executeKernel(timer, tf32_mma_g512_kernel, grid_size, block_size, 
                          d_tcLocalBit, d_sparseA2B, d_dataA, d_rowWindowOffset, 
                          d_metcOffset, d_DenseMatB, d_DenseMatC, numNodes, feature_dim);
        }
        return timer.Elapsed() / EXE_TIME;
    }
    
    float adpBalanceCompute(
        MAT_PTR_TYPE* d_adp_group_offset, 
        MAT_PTR_TYPE* d_tc_offset, 
        MAT_PTR_TYPE* d_adp_row_indices, 
        TCLOCAL_TYPE* d_tcLocalBit, 
        MAT_PTR_TYPE* d_sparseA2B, 
        MAT_VAL_TYPE* d_dataA, 
        MAT_VAL_TYPE* d_DenseMatB, 
        MAT_VAL_TYPE* d_DenseMatC,
        vint numBlocks,
        vint numNodes, 
        vint feature_dim,
        GpuTimer& timer
    ) {
        dim3 grid_size, block_size;
        getKernelConfig(numBlocks, feature_dim, grid_size, block_size);
        
        std::cout << "doing AdpBalance pipelining ..." << std::endl;
        
        if (feature_dim <= THRESHOLD) {
            executeKernel(timer, tf32_mma_balance_kernel, grid_size, block_size,
                          d_adp_group_offset, d_tc_offset, d_adp_row_indices, d_tcLocalBit, 
                          d_sparseA2B, d_dataA, d_DenseMatB, d_DenseMatC,
                          numNodes, feature_dim);
        } else {
            executeKernel(timer, tf32_mma_balance_g128_kernel, grid_size, block_size,
                          d_adp_group_offset, d_tc_offset, d_adp_row_indices, d_tcLocalBit, 
                          d_sparseA2B, d_dataA, d_DenseMatB, d_DenseMatC,
                          numNodes, feature_dim);
        }
        return timer.Elapsed() / EXE_TIME;
    }
};