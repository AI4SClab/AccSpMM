/**
 * @file mma_tf32.cu
 * @author Haisha Zhao
 * @date 2025-04-02
 * 
 * @copyright MIT License (c) 2025 Haisha Zhao
*/

#include "mmio.hpp"
#include "tf32_comp.hpp"

__host__
void tf32_spmm(
    METCFBit<MAT_VAL_TYPE>& metcf_bit,
    vint numNodes, vint numEdges,
    float& elapsed_time,
    const vint feature_dim,
    GpuTimer& timer,
    TF32Compute& tf32_compute
) {
    vint rowWindowOffsetSize    =       metcf_bit.rowWindowOffset.size();
    vint tcOffsetSize           =       metcf_bit.tcOffset.size();
    vint sparseA2BSize          =       metcf_bit.sparseA2B.size();
    vint tcLocalBitSize         =       metcf_bit.tcLocalBit.size();
    vint dataSize               =       metcf_bit.data.size();

    vint numBlocks              =       rowWindowOffsetSize - 1;
    vint denseC_size            =       numBlocks * ROW_WINDOW * feature_dim;
    
    /*---------------- CPU Malloc ------------------*/
    vint* ptr_rowWindowOffset           =   (vint*)malloc(sizeof(vint) * rowWindowOffsetSize);
    vint* ptr_sparseA2B                 =   (vint*)malloc(sizeof(vint) * sparseA2BSize);
    vint* ptr_tcOffset                  =   (vint*)malloc(sizeof(vint) * tcOffsetSize);
    TCLOCAL_TYPE* ptr_tcLocalBit        =   (TCLOCAL_TYPE*)malloc(sizeof(TCLOCAL_TYPE) * tcLocalBitSize);
    MAT_VAL_TYPE* ptr_data              =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * dataSize);
    MAT_VAL_TYPE* DenseMatB             =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * numNodes * feature_dim);
    MAT_VAL_TYPE* DenseMatC             =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * denseC_size);

    std::copy(metcf_bit.rowWindowOffset.begin(), metcf_bit.rowWindowOffset.end(), ptr_rowWindowOffset);
    std::copy(metcf_bit.sparseA2B.begin(), metcf_bit.sparseA2B.end(), ptr_sparseA2B);
    std::copy(metcf_bit.tcOffset.begin(), metcf_bit.tcOffset.end(), ptr_tcOffset);
    std::copy(metcf_bit.tcLocalBit.begin(), metcf_bit.tcLocalBit.end(), ptr_tcLocalBit);
    std::copy(metcf_bit.data.begin(), metcf_bit.data.end(), ptr_data);
    init_vec1(numNodes * feature_dim, DenseMatB, 1.0);
    init_vec1(denseC_size, DenseMatC, 0.0);

    /*---------------- GPU Malloc ------------------*/
    vint* d_rowWindowOffset, *d_sparseA2B, *d_tcOffset;
    MAT_VAL_TYPE* d_data;
    TCLOCAL_TYPE* d_tcLocalBit;
    MAT_VAL_TYPE*   d_DenseMatB, *d_DenseMatC;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_rowWindowOffset, sizeof(vint) * rowWindowOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sparseA2B, sizeof(vint) * sparseA2BSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tcOffset, sizeof(vint) * tcOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tcLocalBit, sizeof(TCLOCAL_TYPE) * tcLocalBitSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, sizeof(MAT_VAL_TYPE) * dataSize));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_DenseMatB, sizeof(MAT_VAL_TYPE) * numNodes * feature_dim));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size));
    /*---------------- CUDA Memcpy -----------------*/
    CHECK_CUDA_ERROR(cudaMemcpy(d_rowWindowOffset, ptr_rowWindowOffset, sizeof(vint) * rowWindowOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sparseA2B, ptr_sparseA2B, sizeof(vint) * sparseA2BSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tcOffset, ptr_tcOffset, sizeof(vint) * tcOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tcLocalBit, ptr_tcLocalBit, sizeof(TCLOCAL_TYPE) * tcLocalBitSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, ptr_data, sizeof(MAT_VAL_TYPE) * dataSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_DenseMatB, DenseMatB, sizeof(MAT_VAL_TYPE) * numNodes * feature_dim, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_DenseMatC, DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyHostToDevice));

    elapsed_time = tf32_compute.compute(
        d_tcLocalBit, d_sparseA2B, d_data, d_DenseMatB, d_DenseMatC, 
        d_rowWindowOffset, d_tcOffset,
        numNodes, numBlocks, feature_dim, timer
    );

    cudaMemcpy(DenseMatC, d_DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyDeviceToHost);

    free(ptr_rowWindowOffset);
    free(ptr_sparseA2B);
    free(ptr_tcOffset);
    free(ptr_tcLocalBit);
    free(ptr_data);
    free(DenseMatB);
    free(DenseMatC);

    cudaFree(d_rowWindowOffset);
    cudaFree(d_sparseA2B);
    cudaFree(d_tcOffset);
    cudaFree(d_tcLocalBit);
    cudaFree(d_data);
    cudaFree(d_DenseMatB);
    cudaFree(d_DenseMatC);
}

__host__
void adp_tf32_spmm(
    AdpBME<MAT_VAL_TYPE>& adpbme,
    vint numNodes, vint numEdges,
    float& elapsed_time,
    const vint feature_dim,
    GpuTimer& timer,
    TF32Compute& tf32_compute
) {
    vint groupOffsetSize    =   adpbme.groupOffset.size();
    vint tcOffsetSize       =   adpbme.tcOffset.size();
    vint rowIndicesSize     =   adpbme.rowIndices.size();
    vint sparseA2BSize      =   adpbme.sparseA2B.size();
    vint tcLocalBitSize     =   adpbme.tcLocalBit.size();
    vint dataSize           =   adpbme.data.size();

    vint numBlocks          =   groupOffsetSize - 1;
    vint denseC_size        =   std::max(numBlocks * ROW_WINDOW * feature_dim, (adpbme.rowIndices.back() + 8) * feature_dim);

    /*---------------- CPU Malloc ------------------*/
    vint* ptr_groupOffset           =   (vint*)malloc(sizeof(vint) * groupOffsetSize);
    vint* ptr_tcOffset              =   (vint*)malloc(sizeof(vint) * tcOffsetSize);
    vint* ptr_rowIndices            =   (vint*)malloc(sizeof(vint) * rowIndicesSize);
    vint* ptr_sparseA2B             =   (vint*)malloc(sizeof(vint) * sparseA2BSize);
    TCLOCAL_TYPE* ptr_tcLocalBit    =   (TCLOCAL_TYPE*)malloc(sizeof(TCLOCAL_TYPE) * tcLocalBitSize);
    MAT_VAL_TYPE* ptr_data          =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * dataSize);
    MAT_VAL_TYPE* DenseMatB         =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * numNodes * feature_dim);
    MAT_VAL_TYPE* DenseMatC         =   (MAT_VAL_TYPE*)malloc(sizeof(MAT_VAL_TYPE) * denseC_size);

    std::copy(adpbme.groupOffset.begin(), adpbme.groupOffset.end(), ptr_groupOffset);
    std::copy(adpbme.tcOffset.begin(), adpbme.tcOffset.end(), ptr_tcOffset);
    std::copy(adpbme.rowIndices.begin(), adpbme.rowIndices.end(), ptr_rowIndices);
    std::copy(adpbme.sparseA2B.begin(), adpbme.sparseA2B.end(), ptr_sparseA2B);
    std::copy(adpbme.tcLocalBit.begin(), adpbme.tcLocalBit.end(), ptr_tcLocalBit);
    std::copy(adpbme.data.begin(), adpbme.data.end(), ptr_data);

    init_vec1(numNodes * feature_dim, DenseMatB, 1.0);
    init_vec1(denseC_size, DenseMatC, 0.0);

    /*---------------- GPU Malloc ------------------*/
    vint* d_groupOffset, *d_tcOffset, *d_rowIndices, *d_sparseA2B;
    TCLOCAL_TYPE* d_tcLocalBit;
    MAT_VAL_TYPE* d_data, *d_DenseMatB, *d_DenseMatC;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_groupOffset, sizeof(vint) * groupOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tcOffset, sizeof(vint) * tcOffsetSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_rowIndices, sizeof(vint) * rowIndicesSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_sparseA2B, sizeof(vint) * sparseA2BSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_tcLocalBit, sizeof(TCLOCAL_TYPE) * tcLocalBitSize));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_data, sizeof(MAT_VAL_TYPE) * dataSize));

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_DenseMatB, sizeof(MAT_VAL_TYPE) * numNodes * feature_dim));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size));
    /*---------------- CUDA Memcpy -----------------*/
    CHECK_CUDA_ERROR(cudaMemcpy(d_groupOffset, ptr_groupOffset, sizeof(vint) * groupOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tcOffset, ptr_tcOffset, sizeof(vint) * tcOffsetSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_rowIndices, ptr_rowIndices, sizeof(vint) * rowIndicesSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sparseA2B, ptr_sparseA2B, sizeof(vint) * sparseA2BSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_tcLocalBit, ptr_tcLocalBit, sizeof(TCLOCAL_TYPE) * tcLocalBitSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_data, ptr_data, sizeof(MAT_VAL_TYPE) * dataSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_DenseMatB, DenseMatB, sizeof(MAT_VAL_TYPE) * numNodes * feature_dim, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_DenseMatC, DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyHostToDevice));

    elapsed_time = tf32_compute.adpBalanceCompute(
        d_groupOffset, d_tcOffset, d_rowIndices, d_tcLocalBit, 
        d_sparseA2B, d_data, d_DenseMatB, d_DenseMatC,
        numBlocks, numNodes, 
        feature_dim, timer
    );
    cudaMemcpy(DenseMatC, d_DenseMatC, sizeof(MAT_VAL_TYPE) * denseC_size, cudaMemcpyDeviceToHost);

    free(ptr_groupOffset);
    free(ptr_tcOffset);
    free(ptr_rowIndices);
    free(ptr_sparseA2B);
    free(ptr_tcLocalBit);
    free(ptr_data);
    free(DenseMatB);
    free(DenseMatC);

    cudaFree(d_groupOffset);
    cudaFree(d_tcOffset);
    cudaFree(d_rowIndices);
    cudaFree(d_sparseA2B);
    cudaFree(d_tcLocalBit);
    cudaFree(d_data);
    cudaFree(d_DenseMatB);
    cudaFree(d_DenseMatC);
}

__host__
int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: ./mma_tf32 <matrix.mtx> <feature_dim>\n");
        return ERROR_ARGS;
    }
    
    const char* filename = argv[1];
    const int feature_dim = atoi(argv[2]);
        
    const std::string mtx_name = match_filename(std::string(filename));
    
    // Load original matrix
    COO<MAT_VAL_TYPE>* coo = (COO<MAT_VAL_TYPE>*)malloc(sizeof(COO<MAT_VAL_TYPE>));
    int read_status = read_from_mtx<MAT_VAL_TYPE>(filename, coo);
    if(read_status != SUCCESS) {
        return ERROR_READ_MTX;
    }
    
    CSR<MAT_VAL_TYPE> csr = CSR<MAT_VAL_TYPE>(coo);
    
    METCFBit<MAT_VAL_TYPE> metcf_bit;
    metcf_bit.convertFromCSR(csr);
    vint numNodes = coo->rows;
    vint numEdges = coo->nnz;
    
    bool load_balance = balanceStrategy(metcf_bit, 
                                        metcf_bit.tcOffset.size() - 1, 
                                        metcf_bit.rowWindowOffset.size() - 1);
    float elapsed_time = 0.0, spmm_throughput = 0.0;
    
    GpuTimer timer;
    TF32Compute tf32_compute;

    if(load_balance) {
        // Adaptive balance Tensor core operations
        AdpBME<MAT_VAL_TYPE> adpbme;
        adpbme.CSR2AdpBME(csr, 3, 1, 2);
        TF32Compute tf32_compute(10, 10, 128);
        adp_tf32_spmm(adpbme, numNodes, numEdges, elapsed_time, feature_dim, timer, tf32_compute);
            
    } else {
        // Tensor core operations
        TF32Compute tf32_compute;
        tf32_spmm(metcf_bit, numNodes, numEdges, elapsed_time, feature_dim, timer, tf32_compute);
    }
    spmm_throughput = (float(numEdges) * float(feature_dim) * 2.0 * 1000.) 
                    / (elapsed_time * 1000. * 1000. * 1000.);

    std::ofstream outFile("result.csv", std::ios::app);
    if (!outFile) {
        std::cerr << "Error Opening result.csv" << std::endl;
    }
    outFile << mtx_name << ","<< feature_dim << "," << elapsed_time << "," << spmm_throughput << "\n";
    outFile.close();  

    free(coo);
    return 0;
}
