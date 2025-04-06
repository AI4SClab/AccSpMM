/**
 * @file mma_tf32.cuh
 * @author Haisha Zhao
 * @date 2025-04-02
 * 
 * @copyright MIT License (c) 2025 Haisha Zhao
*/

#pragma once
#include "common.hpp"
#include "ptx_tf32.hpp"

__global__
void tf32_mma_kernel(
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const vint*         __restrict__    d_sparseA2B,
    const MAT_VAL_TYPE* __restrict__    d_valueA,
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numNodes,
    const vint feature_dim
) {
    using ARegisters = MAT_VAL_TYPE[2]; 
    using BRegisters = MAT_VAL_TYPE[4]; 
    using CRegisters = MAT_VAL_TYPE[2][4];
    ARegisters fragA;
    BRegisters fragB00;
    BRegisters fragB01;
    BRegisters fragB10;
    BRegisters fragB11;
    CRegisters fragC = {0.0};

    vint bid                  =   blockIdx.x;
    vint offY                 =   (blockIdx.y << 7);
    const vint laneid         =   31 & threadIdx.x;
    const vint warpSize       =   32;
    const vint tid            =   threadIdx.y * warpSize + laneid; 
    const vint local_warpID   =   threadIdx.y;

    vint groupID         =   laneid >> 2;
    vint tID_in_group    =   3 & laneid;

    vint rowA            =   groupID;
    vint colA0           =   tID_in_group;
    vint colA1           =   tID_in_group + 4;

    vint colB02          =   groupID + (local_warpID << 5);
    vint colB13          =   groupID + 8 + (local_warpID << 5);
    vint row01           =   tID_in_group;
    vint row23           =   tID_in_group + 4;
    
    constexpr const int inst_k  = 8;
    constexpr const int inst_n  = 8;

    const vint mat_len = 64;
    const vint idx_len = 8;
    vint  local_idx    = 0;

    __shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
    __shared__ vint         d_sharedSparseA2B[2 * idx_len];
    
    vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
    vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
    
    MAT_PTR_TYPE start_blk_idx  = d_block2Idx[bid];
    MAT_PTR_TYPE end_blk_idx    = d_block2Idx[bid+1];

    /* === pre loop === */
    if(tid < mat_len) {  
        TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx];
        vint start_dataIdx         = d_data2Idx[start_blk_idx];
        if(present_local & (1ULL << tid))
            local_idx = __popcll(present_local << (63 - tid));
        // prefetch 1 tc_block
        if(local_idx == 0) d_sharedSparseA[tid] = 0.0;
        else d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
    }
    // prefetch A2B idx
    if(tid < inst_k) {
        d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid);
        if(start_blk_idx + 1 < end_blk_idx) {
            d_sharedSparseA2B[tid + 8] = load_int_from_global(d_sparseA2B + (start_blk_idx + 1) * inst_k + tid);
        }
    }

    __syncthreads();

    vint dense_rowIdx01 = d_sparseA2B[row01];
    vint dense_rowIdx23 = d_sparseA2B[row23];
    if(dense_rowIdx01 > numNodes) {
        fragB00[0] = 0.0; fragB00[1] = 0.0; 
        fragB01[0] = 0.0; fragB01[1] = 0.0;
    } else {
        vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
        vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
        fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);
        fragB00[1] = load_fp32_from_global(d_MatB + sourceIdx1);
        fragB01[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
        fragB01[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
    }
    if(dense_rowIdx23 > numNodes) {
        fragB00[2] = 0.0; fragB00[3] = 0.0; 
        fragB01[2] = 0.0; fragB01[3] = 0.0;
    } else {
        vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
        vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
        fragB00[2] = load_fp32_from_global(d_MatB + sourceIdx0);
        fragB00[3] = load_fp32_from_global(d_MatB + sourceIdx1);
        fragB01[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
        fragB01[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
    } 

    /* === main loop === */
    for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
        vint sel_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 6;   // 0->iter1
        vint sel_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 6;  // 1->iter1
        vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;
        vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;

        vint dense_rowIdx101 = d_sharedSparseA2B[sel_idx_shm_next + row01];
        vint dense_rowIdx123 = d_sharedSparseA2B[sel_idx_shm_next + row23];
        if(sel_shm_next) {
            if(dense_rowIdx101 > numNodes) {
                fragB10[0] = 0.0; fragB10[1] = 0.0; 
                fragB11[0] = 0.0; fragB11[1] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
                fragB10[0] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB10[1] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB11[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB11[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
            }
            if(dense_rowIdx123 > numNodes) {
                fragB10[2] = 0.0; fragB10[3] = 0.0; 
                fragB11[2] = 0.0; fragB11[3] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
                fragB10[2] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB10[3] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB11[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB11[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
            }
        } else {
            if(dense_rowIdx101 > numNodes) {
                fragB00[0] = 0.0; fragB00[1] = 0.0; 
                fragB01[0] = 0.0; fragB01[1] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
                fragB00[0] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB00[1] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB01[0] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB01[1] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
            }
            if(dense_rowIdx123 > numNodes) {
                fragB00[2] = 0.0; fragB00[3] = 0.0; 
                fragB01[2] = 0.0; fragB01[3] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
                fragB00[2] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB00[3] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB01[2] = load_fp32_from_global(d_MatB + sourceIdx0 + COL_WINDOW_R);
                fragB01[3] = load_fp32_from_global(d_MatB + sourceIdx1 + COL_WINDOW_R);
            }
        }         

        /* === START ASYNC COPY === */
        local_idx = 0;
        if(tid < mat_len) {  
            TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
            vint         start_dataIdx = d_data2Idx[tc_block];
            if(present_local & (1ULL << tid))
                local_idx = __popcll(present_local << (63 - tid));
            if(local_idx == 0) d_sharedSparseA[sel_shm_next + tid] = 0.0;
            else async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
        }
        if(tid < inst_k) {
            if(tc_block + 1 < end_blk_idx)
                async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
        }
        /* === END OF ASYNC COPY === */
        // fetch A
        fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
        fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

        if(sel_shm_next) {
            tf32_m16n8k8(fragB00, fragA, fragC[0]);
            tf32_m16n8k8(fragB01, fragA, fragC[1]);
        } else {
            tf32_m16n8k8(fragB10, fragA, fragC[0]);
            tf32_m16n8k8(fragB11, fragA, fragC[1]);
        }

        wait_group();
		__syncthreads();
    }

    /* === end loop === */
    vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
    fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
    fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

    if(!smem_sel) {
        tf32_m16n8k8(fragB00, fragA, fragC[0]);
        tf32_m16n8k8(fragB01, fragA, fragC[1]);
    } else {
        tf32_m16n8k8(fragB10, fragA, fragC[0]);
        tf32_m16n8k8(fragB11, fragA, fragC[1]);
    }
    
    vint colC  =  0;
    vint rowC  =  0;
    
    vint outOff = (bid << 3) * feature_dim + (local_warpID << 5) + offY;
    #pragma unroll
    for(vint i = 0; i < 4; ++i) {
        rowC = (tID_in_group << 1) + (i & 0x1);
        if(i < 2) colC = groupID;
        else colC = groupID + 8;
        store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
        store_fp32_to_global(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
    }
}

__global__
void tf32_mma_g512_kernel (
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const vint*         __restrict__    d_sparseA2B,
    const MAT_VAL_TYPE* __restrict__    d_valueA,
    const MAT_PTR_TYPE* __restrict__    d_block2Idx,
    const MAT_PTR_TYPE* __restrict__    d_data2Idx,
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE* d_MatC,
    const vint numNodes,
    const vint feature_dim
) {
    using ARegisters = MAT_VAL_TYPE[2];
    using BRegisters = MAT_VAL_TYPE[4];
    using CRegisters = MAT_VAL_TYPE[4][4];
    ARegisters fragA;
    BRegisters fragB00;
    BRegisters fragB01;
    BRegisters fragB02;
    BRegisters fragB03;
    BRegisters fragB10;
    BRegisters fragB11;
    BRegisters fragB12;
    BRegisters fragB13;
    CRegisters fragC = {0.0};

    vint bid                  =   blockIdx.x;
    vint offY                 =   (blockIdx.y << 7);
    const vint laneid         =   31 & threadIdx.x;
    const vint dimTileNum     =   feature_dim / (COL_WINDOW_R << 2);
    const vint tid            =   (threadIdx.y << 5) + laneid;
    const vint local_warpID   =   threadIdx.y;

    vint groupID         =   laneid >> 2;
    vint tID_in_group    =   3 & laneid;

    vint rowA            =   groupID;
    vint colA0           =   tID_in_group;
    vint colA1           =   tID_in_group + 4;

    vint colB02          =   (groupID << 2) + (local_warpID << 6);
    vint colB13          =   (groupID << 2) + (local_warpID << 6) + 8;
    vint row01           =   tID_in_group;
    vint row23           =   tID_in_group + 4;
    
    vint colC  =  0;
    vint rowC  =  0;
    
    constexpr const int inst_k  = 8;
    constexpr const int inst_n  = 8;

    const vint mat_len = 64;
    const vint idx_len = 8;
    vint  local_idx    = 0;

    __shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
    __shared__ vint         d_sharedSparseA2B[2 * idx_len];
    
    vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
    vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
    
    MAT_PTR_TYPE start_blk_idx  = d_block2Idx[bid];
    MAT_PTR_TYPE end_blk_idx    = d_block2Idx[bid+1];

    /* === pre loop === */
    if(tid < mat_len) {  
        TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx];
        vint start_dataIdx         = d_data2Idx[start_blk_idx];
        if(present_local & (1ULL << tid))
            local_idx = __popcll(present_local << (63 - tid));
        // prefetch 1 tc_block
        if(local_idx == 0) d_sharedSparseA[tid] = 0.0;
        else d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
    }
    // prefetch A2B idx
    if(tid < inst_k) {
        d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid);
        if(start_blk_idx + 1 < end_blk_idx) {
            d_sharedSparseA2B[tid + 8] = load_int_from_global(d_sparseA2B + (start_blk_idx + 1) * inst_k + tid);
        }
    }
    __syncthreads();
    if(local_warpID < dimTileNum) {
        vint dense_rowIdx01 = d_sharedSparseA2B[row01];
        vint dense_rowIdx23 = d_sharedSparseA2B[row23];
        if(dense_rowIdx01 > numNodes) {
            fragB00[0] = 0.0; 
            fragB01[0] = 0.0; 
            fragB02[0] = 0.0;  
            fragB03[0] = 0.0; 

            fragB00[1] = 0.0; 
            fragB01[1] = 0.0;
            fragB02[1] = 0.0;
            fragB03[1] = 0.0;
        } else{
            vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
            vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
            float4 t0 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx0));
            float4 t1 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx1));
            fragB00[0] = t0.x;
            fragB01[0] = t0.y;
            fragB02[0] = t0.w;
            fragB03[0] = t0.z;

            fragB00[1] = t1.x;
            fragB01[1] = t1.y;
            fragB02[1] = t1.w;
            fragB03[1] = t1.z;
        }
        if(dense_rowIdx23 > numNodes) {
            fragB00[2] = 0.0;  
            fragB01[2] = 0.0; 
            fragB02[2] = 0.0;  
            fragB03[2] = 0.0; 

            fragB00[3] = 0.0;
            fragB01[3] = 0.0;
            fragB02[3] = 0.0;
            fragB03[3] = 0.0;
        } else{
            vint sourceIdx2 = dense_rowIdx23 * feature_dim + colB02;
            vint sourceIdx3 = dense_rowIdx23 * feature_dim + colB13;
            float4 t2 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx2));
            float4 t3 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx3));
            fragB00[2] = t2.x;
            fragB01[2] = t2.y;
            fragB02[2] = t2.w;
            fragB03[2] = t2.z;

            fragB00[3] = t3.x;
            fragB01[3] = t3.y;
            fragB02[3] = t3.w;
            fragB03[3] = t3.z;
        }
    }
    __syncthreads();
    
    /* === main loop === */
    for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
        vint sel_idx_shm       =   ((tc_block - start_blk_idx + 1) & 1) << 3;
        vint sel_idx_shm_next  =   ((tc_block - start_blk_idx ) & 1) << 3;
        vint sel_shm           =   sel_idx_shm << 3;
        vint sel_shm_next      =   sel_idx_shm_next << 3; 

        if(local_warpID < dimTileNum) {
            vint dense_rowIdx101 = d_sharedSparseA2B[sel_idx_shm_next + row01];
            vint dense_rowIdx123 = d_sharedSparseA2B[sel_idx_shm_next + row23];
            if(sel_shm_next) {
                if(dense_rowIdx101 > numNodes) {
                    fragB10[0] = 0.0; 
                    fragB11[0] = 0.0; 
                    fragB12[0] = 0.0;  
                    fragB13[0] = 0.0; 

                    fragB10[1] = 0.0; 
                    fragB11[1] = 0.0;
                    fragB12[1] = 0.0;
                    fragB13[1] = 0.0;
                } else{
                    vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
                    vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
                    float4 t0 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx0));
                    float4 t1 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx1));
                    fragB10[0] = t0.x;
                    fragB11[0] = t0.y;
                    fragB12[0] = t0.w;
                    fragB13[0] = t0.z;

                    fragB10[1] = t1.x;
                    fragB11[1] = t1.y;
                    fragB12[1] = t1.w;
                    fragB13[1] = t1.z;
                }
                if(dense_rowIdx123 > numNodes) {
                    fragB10[2] = 0.0;  
                    fragB11[2] = 0.0; 
                    fragB12[2] = 0.0;  
                    fragB13[2] = 0.0; 

                    fragB10[3] = 0.0;
                    fragB11[3] = 0.0;
                    fragB12[3] = 0.0;
                    fragB13[3] = 0.0;
                } else{
                    vint sourceIdx2 = dense_rowIdx123 * feature_dim + colB02;
                    vint sourceIdx3 = dense_rowIdx123 * feature_dim + colB13;
                    float4 t2 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx2));
                    float4 t3 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx3));
                    fragB10[2] = t2.x;
                    fragB11[2] = t2.y;
                    fragB12[2] = t2.w;
                    fragB13[2] = t2.z;

                    fragB10[3] = t3.x;
                    fragB11[3] = t3.y;
                    fragB12[3] = t3.w;
                    fragB13[3] = t3.z;
                }
            } else {
                if(dense_rowIdx101 > numNodes) {
                    fragB00[0] = 0.0; 
                    fragB01[0] = 0.0; 
                    fragB02[0] = 0.0;  
                    fragB03[0] = 0.0; 

                    fragB00[1] = 0.0; 
                    fragB01[1] = 0.0;
                    fragB02[1] = 0.0;
                    fragB03[1] = 0.0;
                } else{
                    vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
                    vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
                    float4 t0 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx0));
                    float4 t1 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx1));
                    fragB00[0] = t0.x;
                    fragB01[0] = t0.y;
                    fragB02[0] = t0.w;
                    fragB03[0] = t0.z;

                    fragB00[1] = t1.x;
                    fragB01[1] = t1.y;
                    fragB02[1] = t1.w;
                    fragB03[1] = t1.z;
                }
                if(dense_rowIdx123 > numNodes) {
                    fragB00[2] = 0.0;  
                    fragB01[2] = 0.0; 
                    fragB02[2] = 0.0;  
                    fragB03[2] = 0.0; 

                    fragB00[3] = 0.0;
                    fragB01[3] = 0.0;
                    fragB02[3] = 0.0;
                    fragB03[3] = 0.0;
                } else {
                    vint sourceIdx2 = dense_rowIdx123 * feature_dim + colB02;
                    vint sourceIdx3 = dense_rowIdx123 * feature_dim + colB13;
                    float4 t2 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx2));
                    float4 t3 = vector_fetch_fp32V4(reinterpret_cast<const float4*>(d_MatB + sourceIdx3));
                    fragB00[2] = t2.x;
                    fragB01[2] = t2.y;
                    fragB02[2] = t2.w;
                    fragB03[2] = t2.z;

                    fragB00[3] = t3.x;
                    fragB01[3] = t3.y;
                    fragB02[3] = t3.w;
                    fragB03[3] = t3.z;
                }
            }
        }
        
        /* === START ASYNC COPY === */
        local_idx = 0;
        if(tid < mat_len) {  
            TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
            vint         start_dataIdx = d_data2Idx[tc_block];
            if(present_local & (1ULL << tid))
                local_idx = __popcll(present_local << (63 - tid));
            if(local_idx == 0) d_sharedSparseA[sel_shm_next + tid] = 0.0;
            else async_copy(saPtr + ((sel_shm_next + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
        }
        if(tid < inst_k) {
            if(tc_block + 1 < end_blk_idx)
                async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
        }
        
        /* === END OF ASYNC COPY === */
        // fetch A
        fragA[0] = d_sharedSparseA[sel_shm + rowA * inst_n + colA0];
        fragA[1] = d_sharedSparseA[sel_shm + rowA * inst_n + colA1];

        if(sel_shm_next) {
            tf32_m16n8k8(fragB00, fragA, fragC[0]);
            tf32_m16n8k8(fragB01, fragA, fragC[1]);
            tf32_m16n8k8(fragB02, fragA, fragC[2]);
            tf32_m16n8k8(fragB03, fragA, fragC[3]);
        } else {
            tf32_m16n8k8(fragB10, fragA, fragC[0]);
            tf32_m16n8k8(fragB11, fragA, fragC[1]);
            tf32_m16n8k8(fragB12, fragA, fragC[2]);
            tf32_m16n8k8(fragB13, fragA, fragC[3]);
        }

        wait_group();
		__syncthreads();
    }

    /* === end loop === */
    vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
    fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
    fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];

    if(!smem_sel) {
        tf32_m16n8k8(fragB00, fragA, fragC[0]);
        tf32_m16n8k8(fragB01, fragA, fragC[1]);
        tf32_m16n8k8(fragB02, fragA, fragC[2]);
        tf32_m16n8k8(fragB03, fragA, fragC[3]);
    } else {
        tf32_m16n8k8(fragB10, fragA, fragC[0]);
        tf32_m16n8k8(fragB11, fragA, fragC[1]);
        tf32_m16n8k8(fragB12, fragA, fragC[2]);
        tf32_m16n8k8(fragB13, fragA, fragC[3]);
    }
    
    if(local_warpID < dimTileNum) {
        vint outOff = (bid << 3) * feature_dim + (local_warpID << 6) + offY;
        #pragma unroll
        for(vint i = 0; i < 4; ++i) {
            rowC = (tID_in_group << 1) + (i & 0x1);
            if(i < 2) colC = groupID << 2;
            else colC = (groupID + 8) << 2;
            vint off = outOff + rowC * feature_dim + colC;
            store_fp32_to_global(d_MatC + off, fragC[0][i]);
            store_fp32_to_global(d_MatC + off + 1, fragC[1][i]);
            store_fp32_to_global(d_MatC + off + 2, fragC[2][i]);
            store_fp32_to_global(d_MatC + off + 3, fragC[3][i]);
        }
    }
}

__global__
void tf32_mma_balance_kernel (
    const MAT_PTR_TYPE* __restrict__    d_group_offset,
    const MAT_PTR_TYPE* __restrict__    d_tc_offset,
    const MAT_PTR_TYPE* __restrict__    d_row_indices,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const vint*         __restrict__    d_sparseA2B,
    const MAT_VAL_TYPE* __restrict__    d_valueA,
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE*                       d_MatC,
    const vint numNodes,
    const vint feature_dim
) {
    using ARegisters = MAT_VAL_TYPE[2];
    using BRegisters = MAT_VAL_TYPE[2][4];
    using CRegisters = MAT_VAL_TYPE[2][4];
    ARegisters fragA;
    BRegisters fragB0, fragB1;
    CRegisters fragC = {0.0};

    vint bid                  =   blockIdx.x;
    vint offY                 =   (blockIdx.y << 7);
    const vint laneid         =   31 & threadIdx.x;
    const vint warpSize       =   32;
    const vint tid            =   threadIdx.y * warpSize + laneid;
    const vint local_warpID   =   threadIdx.y;

    vint groupID         =   laneid >> 2;
    vint tID_in_group    =   3 & laneid;

    vint rowA            =   groupID;
    vint colA0           =   tID_in_group;
    vint colA1           =   tID_in_group + 4;

    vint colB02          =   groupID;
    vint colB13          =   groupID + 8;
    vint row01           =   tID_in_group;
    vint row23           =   tID_in_group + 4;
    
    vint colC  =  0;
    vint rowC  =  0;
    
    constexpr const int inst_k  = 8;
    constexpr const int inst_n  = 8;

    const vint mat_len = 64;
    const vint idx_len = 8;
    vint  local_idx    = 0;

    __shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
    __shared__ vint         d_sharedSparseA2B[2 * idx_len];
    
    vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
    vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
    
    MAT_PTR_TYPE start_blk_idx  = d_group_offset[bid];
    MAT_PTR_TYPE end_blk_idx    = d_group_offset[bid+1];
    MAT_PTR_TYPE start_row_idx  = d_row_indices[start_blk_idx];
    MAT_PTR_TYPE next_row_idx   = d_row_indices[start_blk_idx+1];

    /* === pre loop === */
    if(tid < mat_len) {  
        TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx];
        vint         start_dataIdx = d_tc_offset[start_blk_idx];
        if(present_local & (1ULL << tid))
            local_idx = __popcll(present_local << (63 - tid));
        // prefetch 1 tc_block
        if(local_idx == 0) d_sharedSparseA[tid] = 0.0;
        else d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
    }
    // prefetch A2B idx
    if(tid < 2 * inst_k) {
        d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid);
    }
    __syncthreads();

    vint dense_rowIdx01 = d_sharedSparseA2B[row01];
    vint dense_rowIdx23 = d_sharedSparseA2B[row23];
    if(dense_rowIdx01 > numNodes) {
        fragB0[0][0] = 0.0; fragB0[0][1] = 0.0; 
        fragB0[1][0] = 0.0; fragB0[1][1] = 0.0;
    } else {
        vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
        vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
        vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
        vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
        fragB0[0][0] = load_fp32_from_global(d_MatB + sourceIdx0);
        fragB0[0][1] = load_fp32_from_global(d_MatB + sourceIdx1);
        fragB0[1][0] = load_fp32_from_global(d_MatB + sourceIdx2);
        fragB0[1][1] = load_fp32_from_global(d_MatB + sourceIdx3);
    }
    if(dense_rowIdx23 > numNodes) {
        fragB0[0][2] = 0.0; fragB0[0][3] = 0.0; 
        fragB0[1][2] = 0.0; fragB0[1][3] = 0.0;
    } else {
        vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
        vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
        vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
        vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
        fragB0[0][2] = load_fp32_from_global(d_MatB + sourceIdx0);
        fragB0[0][3] = load_fp32_from_global(d_MatB + sourceIdx1);
        fragB0[1][2] = load_fp32_from_global(d_MatB + sourceIdx2);
        fragB0[1][3] = load_fp32_from_global(d_MatB + sourceIdx3);
    } 

    /* === main loop === */
    for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
        vint shared_mem_sel      = (tc_block - start_blk_idx + 1) & 1;
        vint shared_mem_sel_next = (tc_block - start_blk_idx) & 1;
        vint sel_idx_shm         = ((tc_block - start_blk_idx + 1) & 1) << 3;
        vint sel_idx_shm_next    = ((tc_block - start_blk_idx ) & 1) << 3;

        vint dense_rowIdx101 = d_sharedSparseA2B[sel_idx_shm_next + row01];
        vint dense_rowIdx123 = d_sharedSparseA2B[sel_idx_shm_next + row23];
        if(sel_idx_shm_next) {
            if(dense_rowIdx101 > numNodes) {
                fragB1[0][0] = 0.0; fragB1[0][1] = 0.0; 
                fragB1[1][0] = 0.0; fragB1[1][1] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
                vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
                vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
                fragB1[0][0] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB1[0][1] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB1[1][0] = load_fp32_from_global(d_MatB + sourceIdx2);
                fragB1[1][1] = load_fp32_from_global(d_MatB + sourceIdx3);
            }
            if(dense_rowIdx123 > numNodes) {
                fragB1[0][2] = 0.0; fragB1[0][3] = 0.0; 
                fragB1[1][2] = 0.0; fragB1[1][3] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
                vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
                vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
                fragB1[0][2] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB1[0][3] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB1[1][2] = load_fp32_from_global(d_MatB + sourceIdx2);
                fragB1[1][3] = load_fp32_from_global(d_MatB + sourceIdx3);
            } 
        } else {
            if(dense_rowIdx101 > numNodes) {
                fragB0[0][0] = 0.0; fragB0[0][1] = 0.0; 
                fragB0[1][0] = 0.0; fragB0[1][1] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx101 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx101 * feature_dim + colB13;
                vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
                vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
                fragB0[0][0] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB0[0][1] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB0[1][0] = load_fp32_from_global(d_MatB + sourceIdx2);
                fragB0[1][1] = load_fp32_from_global(d_MatB + sourceIdx3);
            }
            if(dense_rowIdx123 > numNodes) {
                fragB0[0][2] = 0.0; fragB0[0][3] = 0.0; 
                fragB0[1][2] = 0.0; fragB0[1][3] = 0.0;
            } else {
                vint sourceIdx0 = dense_rowIdx123 * feature_dim + colB02;
                vint sourceIdx1 = dense_rowIdx123 * feature_dim + colB13;
                vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
                vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
                fragB0[0][2] = load_fp32_from_global(d_MatB + sourceIdx0);
                fragB0[0][3] = load_fp32_from_global(d_MatB + sourceIdx1);
                fragB0[1][2] = load_fp32_from_global(d_MatB + sourceIdx2);
                fragB0[1][3] = load_fp32_from_global(d_MatB + sourceIdx3);
            } 
        }

        /* === START ASYNC COPY === */
        local_idx = 0;
        if(tid < mat_len) {  
            TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
            vint         start_dataIdx = d_tc_offset[tc_block];
            if(present_local & (1ULL << tid))
                local_idx = __popcll(present_local << (63 - tid));
            if(local_idx == 0) d_sharedSparseA[(shared_mem_sel_next << 6) + tid] = 0.0;
            else async_copy(saPtr + (((shared_mem_sel_next << 6) + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
        }
        if(tid < inst_k) {
            if(tc_block + 1 < end_blk_idx)
                async_copy_idx(siPtr + ((sel_idx_shm + tid) << 2), d_sparseA2B + (tc_block + 1) * inst_k + tid);
        }
        /* === END OF ASYNC COPY === */
        // fetch A
        fragA[0] = d_sharedSparseA[(shared_mem_sel << 6) + rowA * inst_n + colA0];
        fragA[1] = d_sharedSparseA[(shared_mem_sel << 6) + rowA * inst_n + colA1];

        if(sel_idx_shm_next) {
            tf32_m16n8k8(fragB0[0], fragA, fragC[0]);
            tf32_m16n8k8(fragB0[1], fragA, fragC[1]);
        } else {
            tf32_m16n8k8(fragB1[0], fragA, fragC[0]);
            tf32_m16n8k8(fragB1[1], fragA, fragC[1]);
        }

        next_row_idx = d_row_indices[tc_block];

        if(next_row_idx != start_row_idx) {
            vint outOff = start_row_idx * feature_dim + (local_warpID << 5) + offY;
            #pragma unroll
            for(vint i = 0; i < 4; ++i) {
                rowC = (tID_in_group << 1) + (i & 0x1);
                if(i < 2) colC = groupID;
                else colC = groupID + 8;
                atomicAdd(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
                atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
                fragC[0][i] = 0.0;
                fragC[1][i] = 0.0;
            }
        }
        start_row_idx = next_row_idx;

        wait_group();
		__syncthreads();
    }

    /* === end loop === */
    vint smem_sel  = ((end_blk_idx - start_blk_idx + 1) & 1) << 6;
    fragA[0] = d_sharedSparseA[smem_sel + rowA * inst_n + colA0];
    fragA[1] = d_sharedSparseA[smem_sel + rowA * inst_n + colA1];
    
    if(!smem_sel) {
        tf32_m16n8k8(fragB0[0], fragA, fragC[0]);
        tf32_m16n8k8(fragB0[1], fragA, fragC[1]);
    } else {
        tf32_m16n8k8(fragB1[1], fragA, fragC[0]);
        tf32_m16n8k8(fragB1[1], fragA, fragC[1]);
    }

    vint outOff = start_row_idx * feature_dim + (local_warpID << 5) + offY;
    #pragma unroll
    for(vint i = 0; i < 4; ++i) {
        rowC = (tID_in_group << 1) + (i & 0x1);
        if(i < 2) colC = groupID;
        else colC = groupID + 8;
        atomicAdd(d_MatC + outOff + rowC * feature_dim + colC, fragC[0][i]);
        atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC[1][i]);
    }
}

__global__
void tf32_mma_balance_g128_kernel (
    const MAT_PTR_TYPE* __restrict__    d_group_offset,
    const MAT_PTR_TYPE* __restrict__    d_tc_offset,
    const MAT_PTR_TYPE* __restrict__    d_row_indices,
    const TCLOCAL_TYPE* __restrict__    d_tcLocalBit, 
    const vint*         __restrict__    d_sparseA2B,
    const MAT_VAL_TYPE* __restrict__    d_valueA,
    const MAT_VAL_TYPE* __restrict__    d_MatB, 
    MAT_VAL_TYPE*                       d_MatC,
    const vint numNodes,
    const vint feature_dim
) {
    using ARegisters = MAT_VAL_TYPE[2];
    using BRegisters = MAT_VAL_TYPE[2][4];
    using CRegisters = MAT_VAL_TYPE[2][4];
    ARegisters fragA;
    BRegisters fragB0, fragB1;
    CRegisters fragC0 = {0.0}, fragC1 = {0.0};

    vint bid                  =   blockIdx.x;
    vint offY                 =   (blockIdx.y << 7);
    const vint laneid         =   31 & threadIdx.x;
    const vint warpSize       =   32;
    
    const vint tid            =   threadIdx.y * warpSize + laneid;
    const vint local_warpID   =   threadIdx.y;

    vint groupID         =   laneid >> 2;
    vint tID_in_group    =   3 & laneid;

    vint rowA            =   groupID;
    vint colA0           =   tID_in_group;
    vint colA1           =   tID_in_group + 4;

    vint colB02          =   groupID;
    vint colB13          =   groupID + 8;
    vint row01           =   tID_in_group;
    vint row23           =   tID_in_group + 4;
    
    vint colC  =  0;
    vint rowC  =  0;

    constexpr const int inst_k  = 8;
    constexpr const int inst_n  = 8;

    const vint mat_len = 64;
    const vint idx_len = 8;
    vint  local_idx    = 0;

    __shared__ MAT_VAL_TYPE d_sharedSparseA[2 * mat_len];
    __shared__ vint         d_sharedSparseA2B[2 * idx_len];
    
    vint saPtr = __cvta_generic_to_shared(d_sharedSparseA);
    vint siPtr = __cvta_generic_to_shared(d_sharedSparseA2B);
    
    MAT_PTR_TYPE start_blk_idx  = d_group_offset[bid];
    MAT_PTR_TYPE end_blk_idx    = d_group_offset[bid+1];
    MAT_PTR_TYPE start_row_idx  = d_row_indices[start_blk_idx];
    MAT_PTR_TYPE next_row_idx   = d_row_indices[start_blk_idx+1];
    vint         start_dataIdx  = d_tc_offset[start_blk_idx];

    /* === pre loop === */
    if(tid < mat_len) {  
        TCLOCAL_TYPE present_local = d_tcLocalBit[start_blk_idx];
        if(present_local & (1ULL << tid))
            local_idx = __popcll(present_local << (63 - tid));
        // prefetch 1 tc_block
        if(local_idx == 0) d_sharedSparseA[tid] = 0.0;
        else d_sharedSparseA[tid] = load_fp32_from_global2shared(d_valueA + start_dataIdx + local_idx - 1);
    }
    // prefetch A2B idx
    if(tid < inst_k) {
        d_sharedSparseA2B[tid] = load_int_from_global(d_sparseA2B + start_blk_idx * inst_k + tid);
    }
    __syncthreads();

    /* === main loop === */
    for(vint tc_block = start_blk_idx + 1; tc_block < end_blk_idx; ++tc_block) { 
        vint shared_mem_sel      = (tc_block - start_blk_idx + 1) & 1;
        vint shared_mem_sel_next = (tc_block - start_blk_idx) & 1;

        vint dense_rowIdx01 = d_sharedSparseA2B[(shared_mem_sel << 3) + row01];
        vint dense_rowIdx23 = d_sharedSparseA2B[(shared_mem_sel << 3) + row23];
        if(dense_rowIdx01 > numNodes) {
            fragB0[0][0] = 0.0; fragB0[0][1] = 0.0; 
            fragB0[1][0] = 0.0; fragB0[1][1] = 0.0;
            fragB1[0][0] = 0.0; fragB1[0][1] = 0.0; 
            fragB1[1][0] = 0.0; fragB1[1][1] = 0.0;
        } else {
            vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
            vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
            vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
            vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
            fragB0[0][0] = load_fp32_from_global(d_MatB + sourceIdx0);
            fragB0[0][1] = load_fp32_from_global(d_MatB + sourceIdx1);
            fragB0[1][0] = load_fp32_from_global(d_MatB + sourceIdx2);
            fragB0[1][1] = load_fp32_from_global(d_MatB + sourceIdx3);

            vint sourceIdx4 = sourceIdx2 + COL_WINDOW_R;
            vint sourceIdx5 = sourceIdx3 + COL_WINDOW_R;
            vint sourceIdx6 = sourceIdx4 + COL_WINDOW_R;
            vint sourceIdx7 = sourceIdx5 + COL_WINDOW_R;
            fragB1[0][0] = load_fp32_from_global(d_MatB + sourceIdx4); 
            fragB1[0][1] = load_fp32_from_global(d_MatB + sourceIdx5); 
            fragB1[1][0] = load_fp32_from_global(d_MatB + sourceIdx6); 
            fragB1[1][1] = load_fp32_from_global(d_MatB + sourceIdx7); 
        }
        if(dense_rowIdx23 > numNodes) {
            fragB0[0][2] = 0.0; fragB0[0][3] = 0.0; 
            fragB0[1][2] = 0.0; fragB0[1][3] = 0.0;
            fragB1[0][2] = 0.0; fragB1[0][3] = 0.0; 
            fragB1[1][2] = 0.0; fragB1[1][3] = 0.0;
        } else {
            vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
            vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
            vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
            vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
            fragB0[0][2] = load_fp32_from_global(d_MatB + sourceIdx0);
            fragB0[0][3] = load_fp32_from_global(d_MatB + sourceIdx1);
            fragB0[1][2] = load_fp32_from_global(d_MatB + sourceIdx2);
            fragB0[1][3] = load_fp32_from_global(d_MatB + sourceIdx3);

            vint sourceIdx4 = sourceIdx2 + COL_WINDOW_R;
            vint sourceIdx5 = sourceIdx3 + COL_WINDOW_R;
            vint sourceIdx6 = sourceIdx4 + COL_WINDOW_R;
            vint sourceIdx7 = sourceIdx5 + COL_WINDOW_R;
            fragB1[0][2] = load_fp32_from_global(d_MatB + sourceIdx4); 
            fragB1[0][3] = load_fp32_from_global(d_MatB + sourceIdx5);
            fragB1[1][2] = load_fp32_from_global(d_MatB + sourceIdx6); 
            fragB1[1][3] = load_fp32_from_global(d_MatB + sourceIdx7);
        } 

        /* === START ASYNC COPY === */
        start_dataIdx = d_tc_offset[tc_block];
        local_idx = 0;
        if(tid < mat_len) {  
            TCLOCAL_TYPE present_local = d_tcLocalBit[tc_block];
            if(present_local & (1ULL << tid))
                local_idx = __popcll(present_local << (63 - tid));
            if(local_idx == 0) d_sharedSparseA[(shared_mem_sel_next << 6) + tid] = 0.0;
            else async_copy(saPtr + (((shared_mem_sel_next << 6) + tid) << 2), d_valueA + start_dataIdx + local_idx - 1);
        }
        if(tid < inst_k) {
            async_copy_idx(siPtr + (((shared_mem_sel_next << 3) + tid) << 2), d_sparseA2B + tc_block * inst_k + tid);
        }
        /* === END OF ASYNC COPY === */
        // fetch A
        fragA[0] = d_sharedSparseA[(shared_mem_sel << 6) + rowA * inst_n + colA0];
        fragA[1] = d_sharedSparseA[(shared_mem_sel << 6) + rowA * inst_n + colA1];
        
        tf32_m16n8k8(fragB0[0], fragA, fragC0[0]);
        tf32_m16n8k8(fragB0[1], fragA, fragC0[1]);
        tf32_m16n8k8(fragB1[0], fragA, fragC1[0]);
        tf32_m16n8k8(fragB1[1], fragA, fragC1[1]);

        next_row_idx = d_row_indices[tc_block];

        if(next_row_idx != start_row_idx) {
            vint outOff = start_row_idx * feature_dim + (local_warpID << 6) + offY;
            #pragma unroll
            for(vint i = 0; i < 4; ++i) {
                rowC = (tID_in_group << 1) + (i & 0x1);
                if(i < 2) colC = groupID;
                else colC = groupID + 8;
                atomicAdd(d_MatC + outOff + rowC * feature_dim + colC, fragC0[0][i]);
                atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC0[1][i]);
                atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R * 2, fragC1[0][i]);
                atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R * 3, fragC1[1][i]);
                fragC0[0][i] = 0.0; fragC0[1][i] = 0.0; fragC1[0][i] = 0.0; fragC1[1][i] = 0.0;
            }
        }
        start_row_idx = next_row_idx;

        wait_group();
		__syncthreads();
    }

    /* === end loop === */
    vint smem_sel  = (end_blk_idx - start_blk_idx + 1) & 1;
    fragA[0] = d_sharedSparseA[(smem_sel << 6) + rowA * inst_n + colA0];
    fragA[1] = d_sharedSparseA[(smem_sel << 6) + rowA * inst_n + colA1];

    vint dense_rowIdx01 = d_sharedSparseA2B[(smem_sel << 3) + row01];
    vint dense_rowIdx23 = d_sharedSparseA2B[(smem_sel << 3) + row23];
    if(dense_rowIdx01 > numNodes) {
        fragB0[0][0] = 0.0; fragB0[0][1] = 0.0; 
        fragB0[1][0] = 0.0; fragB0[1][1] = 0.0; 
        fragB1[0][0] = 0.0; fragB1[0][1] = 0.0; 
        fragB1[1][0] = 0.0; fragB1[1][1] = 0.0; 
    } else {
        vint sourceIdx0 = dense_rowIdx01 * feature_dim + colB02;
        vint sourceIdx1 = dense_rowIdx01 * feature_dim + colB13;
        vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
        vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
        fragB0[0][0] = load_fp32_from_global(d_MatB + sourceIdx0);
        fragB0[0][1] = load_fp32_from_global(d_MatB + sourceIdx1);
        fragB0[1][0] = load_fp32_from_global(d_MatB + sourceIdx2);
        fragB0[1][1] = load_fp32_from_global(d_MatB + sourceIdx3);
        vint sourceIdx4 = sourceIdx2 + COL_WINDOW_R;
        vint sourceIdx5 = sourceIdx3 + COL_WINDOW_R;
        vint sourceIdx6 = sourceIdx4 + COL_WINDOW_R;
        vint sourceIdx7 = sourceIdx5 + COL_WINDOW_R;
        fragB1[0][0] = load_fp32_from_global(d_MatB + sourceIdx4); 
        fragB1[0][1] = load_fp32_from_global(d_MatB + sourceIdx5); 
        fragB1[1][0] = load_fp32_from_global(d_MatB + sourceIdx6); 
        fragB1[1][1] = load_fp32_from_global(d_MatB + sourceIdx7); 
    }
    if(dense_rowIdx23 > numNodes) {
        fragB0[0][2] = 0.0; fragB0[0][3] = 0.0;
        fragB0[1][2] = 0.0; fragB0[1][3] = 0.0;
        fragB1[0][2] = 0.0; fragB1[0][3] = 0.0;
        fragB1[1][2] = 0.0; fragB1[1][3] = 0.0;
    } else {
        vint sourceIdx0 = dense_rowIdx23 * feature_dim + colB02;
        vint sourceIdx1 = dense_rowIdx23 * feature_dim + colB13;
        vint sourceIdx2 = sourceIdx0 + COL_WINDOW_R;
        vint sourceIdx3 = sourceIdx1 + COL_WINDOW_R;
        fragB0[0][2] = load_fp32_from_global(d_MatB + sourceIdx0);
        fragB0[0][3] = load_fp32_from_global(d_MatB + sourceIdx1);
        fragB0[1][2] = load_fp32_from_global(d_MatB + sourceIdx2);
        fragB0[1][3] = load_fp32_from_global(d_MatB + sourceIdx3);
        vint sourceIdx4 = sourceIdx2 + COL_WINDOW_R;
        vint sourceIdx5 = sourceIdx3 + COL_WINDOW_R;
        vint sourceIdx6 = sourceIdx4 + COL_WINDOW_R;
        vint sourceIdx7 = sourceIdx5 + COL_WINDOW_R;
        fragB1[0][2] = load_fp32_from_global(d_MatB + sourceIdx4); 
        fragB1[0][3] = load_fp32_from_global(d_MatB + sourceIdx5);
        fragB1[1][2] = load_fp32_from_global(d_MatB + sourceIdx6); 
        fragB1[1][3] = load_fp32_from_global(d_MatB + sourceIdx7);
    }        

    tf32_m16n8k8(fragB0[0], fragA, fragC0[0]);
    tf32_m16n8k8(fragB0[1], fragA, fragC0[1]);
    tf32_m16n8k8(fragB1[0], fragA, fragC1[0]);
    tf32_m16n8k8(fragB1[1], fragA, fragC1[1]);

    vint outOff = start_row_idx * feature_dim + (local_warpID << 6) + offY;
    #pragma unroll
    for(vint i = 0; i < 4; ++i) {
        rowC = (tID_in_group << 1) + (i & 0x1);
        if(i < 2) colC = groupID;
        else colC = groupID + 8;
        atomicAdd(d_MatC + outOff + rowC * feature_dim + colC, fragC0[0][i]);
        atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R, fragC0[1][i]);
        atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R * 2, fragC1[0][i]);
        atomicAdd(d_MatC + outOff + rowC * feature_dim + colC + COL_WINDOW_R * 3, fragC1[1][i]);
    }
}