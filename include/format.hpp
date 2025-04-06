/**
 * @file format.hpp
 * @author Haisha Zhao
 * @date 2025-04-02
 * 
 * @copyright MIT License (c) 2025 Haisha Zhao
*/

#pragma once
#include "common.hpp"

template <typename DataType>
class SparseMatrixFormat {
public:
    vint rows, cols, nnz;
    
    virtual void show() const = 0;
    virtual long long printSize(bool print = true) const = 0;
    virtual ~SparseMatrixFormat() = default;
    
protected:
    // Common utility function for all formats
    void padListLength(std::vector<vint>& lst, int alignment = COL_WINDOW) const {
        vint current_length = lst.size();
        vint remainder = current_length % alignment;
        if (remainder != 0) {
            vint padding_size = alignment - remainder;
            lst.insert(lst.end(), padding_size, UINT32_MAX);
        }
    }
};

template <typename DataType>
class COO : public SparseMatrixFormat<DataType> {
public:
    using SparseMatrixFormat<DataType>::rows;
    using SparseMatrixFormat<DataType>::cols;
    using SparseMatrixFormat<DataType>::nnz;
    
    vint* row;
    vint* col;
    DataType* data;
    
    COO() : row(nullptr), col(nullptr), data(nullptr) {}
    
    ~COO() override {
        if (row)  free(row);
        if (col)  free(col);
        if (data) free(data);
    }
    
    void show() const override {
        for (vint i = 0; i < nnz; ++i) {
            printf("row = %d, col = %d, data = %lf\n", row[i], col[i], data[i]);
        }
    }
    
    long long printSize(bool print = true) const override {
        long long size = nnz * (2 * sizeof(vint) + sizeof(DataType));
        if (print) {
            std::cout << "\n========== size of COO =========\n";
            std::cout << "Total size: " << size << " bytes" << std::endl;
        }
        return size;
    }
    
    bool is_sorted_matrix() const {
        for (vint i = 0; i < nnz-1; ++i) {
            if ((row[i] > row[i+1]) || (row[i] == row[i+1] && col[i] >= col[i+1])) {
                printf("not sorted: row: %u, col: %u\n", row[i], col[i]);
                printf("not sorted: row: %u, col: %u\n", row[i+1], col[i+1]);
                return false;
            }
        }
        return true;
    }

    bool sort_matrix() {
        std::vector<std::tuple<vint, vint, DataType>> tuples(nnz);
        for (vint i = 0; i < nnz; ++i) {
            tuples[i] = std::make_tuple(row[i], col[i], data[i]);
        }

        std::sort(tuples.begin(), tuples.end());

        for (vint i = 0; i < nnz; ++i) {
            row[i]  = std::get<0>(tuples[i]);
            col[i]  = std::get<1>(tuples[i]);
            data[i] = std::get<2>(tuples[i]);
        }

        return is_sorted_matrix();
    }
};

template <typename DataType>
class CSR : public SparseMatrixFormat<DataType> {
public:
    using SparseMatrixFormat<DataType>::rows;
    using SparseMatrixFormat<DataType>::cols;
    using SparseMatrixFormat<DataType>::nnz;
    
    std::vector<vint> row_ptr;
    std::vector<vint> col_idx;
    std::vector<DataType> data;
    
    CSR() = default;
    
    // Convert from COO
    explicit CSR(const COO<DataType>* coo) {
        rows = coo->rows;
        cols = coo->cols;
        nnz  = coo->nnz;
        row_ptr.resize(coo->rows + 1, 0);
        col_idx.resize(coo->nnz);
        data.resize(coo->nnz);

        // Count elements per row
        for (vint i = 0; i < coo->nnz; ++i) {
            row_ptr[coo->row[i] + 1]++;
        }
        
        // Calculate cumulative sum for row_ptr
        for (vint i = 0; i < coo->rows; ++i) {
            row_ptr[i+1] += row_ptr[i];
        }

        // Place elements in the correct positions
        for (vint i = 0; i < coo->nnz; ++i) {
            vint row      = coo->row[i];
            vint dest     = row_ptr[row];
            data[dest]    = coo->data[i];
            col_idx[dest] = coo->col[i];
            row_ptr[row]++;
        }

        // Restore row_ptr
        for (vint i = coo->rows; i > 0; --i) {
            row_ptr[i] = row_ptr[i - 1];
        }
        row_ptr[0] = 0;
    }
    
    void show() const override {
        printf("Matrix in CSR format, Row: %d, Col: %d, NNZ: %d\n", rows, cols, nnz);
        for (vint i = 0; i < row_ptr.size() - 1; ++i) {
            printf("\nrow = %d\n", i);
            for (vint j = row_ptr[i]; j < row_ptr[i+1]; ++j) {
                printf("col = %d: data = %lf\n", col_idx[j], data[j]);
            }
        }
    }

    long long printSize(bool print = true) const override {
        long long size = row_ptr.size() * sizeof(vint)
                       + col_idx.size() * sizeof(vint)
                       + data.size() * sizeof(DataType);
                         
        if (print) {
            std::cout << "\n========== size of CSR =========\n";
            std::cout << "Total size: " << size << " bytes" << std::endl;
            std::cout << "row_ptr: " << row_ptr.size() * sizeof(vint) << " bytes" << std::endl;
            std::cout << "col_idx: " << col_idx.size() * sizeof(vint) << " bytes" << std::endl;
            std::cout << "data: " << data.size() * sizeof(DataType) << " bytes" << std::endl;
        }
        
        return size;
    }
};

// Base class for TC formats
template <typename DataType>
class TensorCoreFormat : public SparseMatrixFormat<DataType> {
public:
    std::vector<vint> sparseA2B;
    std::vector<DataType> data;
    
    // Common functionality for tensor core formats
    std::unordered_map<vint, vint> extractUniqueColumns(
        const CSR<DataType>& csr, vint start_row, vint end_row, 
        std::vector<vint>& unique_edges) const {
        
        // Get all column indices for this row window
        std::vector<vint> neighbor_window(csr.col_idx.begin() + start_row, 
                                          csr.col_idx.begin() + end_row);
        
        // Sort and get unique columns
        std::sort(neighbor_window.begin(), neighbor_window.end());
        std::unique_copy(neighbor_window.begin(), neighbor_window.end(), 
                         std::back_inserter(unique_edges));
        
        // Create map for quick lookup
        std::unordered_map<vint, vint> clean_edges2col;
        for (vint i = 0; i < unique_edges.size(); ++i) {
            clean_edges2col[unique_edges[i]] = i;
        }
        
        return clean_edges2col;
    }
};

// METCFBit Format
template <typename DataType>
class METCFBit : public TensorCoreFormat<DataType> {
public:
    using TensorCoreFormat<DataType>::sparseA2B;
    using TensorCoreFormat<DataType>::data;
    
    std::vector<vint> rowWindowOffset;
    std::vector<vint> tcOffset;
    std::vector<TCLOCAL_TYPE> tcLocalBit;
    
    METCFBit() = default;
    
    void convertFromCSR(const CSR<DataType>& csr) {
        vint num_nodes = csr.row_ptr.size() - 1;
        rowWindowOffset.push_back(0);

        vint tc_tmp_len = ROW_WINDOW * COL_WINDOW == 64 ? 1 : 2;

        for (vint iter = 0; iter < num_nodes; iter += ROW_WINDOW) {
            vint windowId = iter / ROW_WINDOW;
            vint block_start = csr.row_ptr[iter];
            vint block_end = csr.row_ptr[std::min(iter + ROW_WINDOW, num_nodes)];

            std::vector<vint> unique_edges;
            std::unordered_map<vint, vint> clean_edges2col = this->extractUniqueColumns(csr, block_start, block_end, unique_edges);

            this->padListLength(unique_edges);
            sparseA2B.insert(sparseA2B.end(), unique_edges.begin(), unique_edges.end());

            vint window_tc_num = (unique_edges.size() + COL_WINDOW - 1) / COL_WINDOW;
            rowWindowOffset.push_back(rowWindowOffset[windowId] + window_tc_num);

            tcOffset.resize(tcOffset.size() + window_tc_num, 0);
            
            std::vector<TCLOCAL_TYPE> tcLocalIdtmp(window_tc_num * tc_tmp_len);
            std::vector<std::vector<DataType>> datatmp(window_tc_num);
            for (vint r = iter; r < std::min(iter + ROW_WINDOW, num_nodes); ++r) {
                for (vint nnz_id = csr.row_ptr[r]; nnz_id < csr.row_ptr[r + 1]; ++nnz_id) {
                    vint c_idx = clean_edges2col[csr.col_idx[nnz_id]];
                    vint offset_index = rowWindowOffset[windowId] + c_idx / COL_WINDOW;
                    tcOffset[offset_index]++;
                    uint8_t local_idx = (r % ROW_WINDOW) * COL_WINDOW + c_idx % COL_WINDOW;

                    if (tc_tmp_len == 1){
                        TCLOCAL_TYPE local_idx_mask = 1ULL << local_idx;
                        tcLocalIdtmp[c_idx / COL_WINDOW] |= local_idx_mask;
                    } else if (tc_tmp_len == 2){
                        if(local_idx < 64){
                            TCLOCAL_TYPE local_idx_mask = 1ULL << local_idx; 
                            tcLocalIdtmp[c_idx / COL_WINDOW * 2] |= local_idx_mask;
                        }else{
                            TCLOCAL_TYPE local_idx_mask = 1ULL << (local_idx - 64);
                            tcLocalIdtmp[c_idx / COL_WINDOW * 2 + 1] |= local_idx_mask;
                        }
                    } else {
                        std::cout << "ERROR: in convertFromSCR(): TC shape is not surported!!!" << std::endl;
                    }

                    datatmp[c_idx / COL_WINDOW].push_back(csr.data[nnz_id]);
                }
            }
     
            tcLocalBit.insert(tcLocalBit.end(), tcLocalIdtmp.begin(), tcLocalIdtmp.end());
            
            for (vint i = 0; i < datatmp.size(); ++i) {
                data.insert(data.end(), datatmp[i].begin(), datatmp[i].end());
            }
        }

        tcOffset.insert(tcOffset.begin(), 0);
        std::partial_sum(tcOffset.begin(), tcOffset.end(), tcOffset.begin());
    }
    
    void show() const override {
        std::cout << "\n========== METCFBit =========\n";
        
        std::cout << "rowWindowOffset:" << std::endl;
        for (auto val : rowWindowOffset) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "tcLocalBit:" << std::endl;
        for (auto val : tcLocalBit) {
            std::cout << std::bitset<64>(val) << std::endl;
            std::cout << val << std::endl;
        }
        std::cout << std::endl;

        std::cout << "sparseA2B:" << std::endl;
        for (auto val : sparseA2B) {
            std::cout << val << " ";
        }
        std::cout << std::endl;

        std::cout << "data:" << std::endl;
        for (auto val : data) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    long long printSize (bool print = true) const override {
        long long s1 = rowWindowOffset.size() * sizeof(int);
        long long s2 = tcOffset.size() * sizeof(int);
        long long s3 = tcLocalBit.size() * sizeof(TCLOCAL_TYPE);
        long long s4 = sparseA2B.size() * sizeof(int);
        long long total = s1 + s2 + s3 + s4;

        if(print){
            std::cout << "\n========== size of ME-CTF-bit =========\n";
            std::cout << "total size(without data): " << total << " bytes" << std::endl;
            std::cout << "rowWindowOffset: " << s1 << " bytes" << std::endl;
            std::cout << "tcOffset: " << s2 << " bytes"  << std::endl;
            std::cout << "tcLocalBit: " << s3 << " bytes"  << std::endl;
            std::cout << "sparseA2B: " << s4 << " bytes"  << std::endl;

            std::cout << std::endl;
        }
        
        return total;
    }
};

template <typename DataType>
class AdpBME : public TensorCoreFormat<DataType> {
public:
    using TensorCoreFormat<DataType>::sparseA2B;
    using TensorCoreFormat<DataType>::data;
    
    std::vector<vint> groupOffset;
    std::vector<vint> tcOffset;
    std::vector<vint> rowIndices;
    std::vector<TCLOCAL_TYPE> tcLocalBit;

    AdpBME() = default;

    void CSR2AdpBME(const CSR<DataType>& csr, vint t_load, vint t_comp, vint t_write, vint target_num = TARGET_NUM) {

        vint num_nodes = csr.rows;
        vint total_tc_num = 0;
        
        // vint t_tc = t_load + t_comp;
        vint target = target_num * (t_load + t_comp) + t_write;
        vint current_group_time = 0;
        vint current_group_size = 0;
        groupOffset.push_back(0);

        for (vint iter = 0; iter < num_nodes; iter += ROW_WINDOW) {
            vint block_start = csr.row_ptr[iter];
            vint block_end = csr.row_ptr[std::min(iter + ROW_WINDOW, num_nodes)];

            std::vector<vint> unique_edges;
            std::unordered_map<vint, vint> clean_edges2col = this->extractUniqueColumns(csr, block_start, block_end, unique_edges);

            this->padListLength(unique_edges);
            sparseA2B.insert(sparseA2B.end(), unique_edges.begin(), unique_edges.end());

            vint window_tc_num = (unique_edges.size() + COL_WINDOW - 1) / COL_WINDOW;
            // vint window_group_num = (window_tc_num + GROUP_LEN - 1) / GROUP_LEN;
            tcOffset.resize(tcOffset.size() + window_tc_num, 0);
            rowIndices.resize(rowIndices.size() + window_tc_num, iter);

            vint additional_time = 0;
            for(vint i = 0; i < window_tc_num; ++i){
                if(i == 0) {
                    additional_time = t_load + t_comp + t_write;
                } else{
                    additional_time = t_load + t_comp;
                }

                if (current_group_size > 0 && current_group_time + additional_time > target) {
                    groupOffset.push_back(current_group_size);
                    current_group_time = 0;
                    current_group_size = 0;
                }

                current_group_time += additional_time;
                current_group_size++;
            }

            std::vector<std::vector<DataType>> datatmp(window_tc_num);
            vint tc_tmp_len = ROW_WINDOW * COL_WINDOW == 64 ? 1 : 2; // ROW_WINDOW * COL_WINDOW = 64 | 128
            std::vector<TCLOCAL_TYPE> tcLocalBittmp(window_tc_num * tc_tmp_len, 0);
            for (vint r = iter; r < std::min(iter + ROW_WINDOW, num_nodes); ++r) {
                for (vint nnz_id = csr.row_ptr[r]; nnz_id < csr.row_ptr[r + 1]; ++nnz_id) {
                    vint c_idx = clean_edges2col[csr.col_idx[nnz_id]];
                    vint tc_idx = total_tc_num + c_idx / COL_WINDOW;
                    tcOffset[tc_idx]++;
                    if (tc_tmp_len == 1){
                        TCLOCAL_TYPE local_idx_mask = 1ULL << ((r % ROW_WINDOW) * COL_WINDOW + c_idx % COL_WINDOW);
                        tcLocalBittmp[c_idx / COL_WINDOW] |= local_idx_mask;
                    } else if(tc_tmp_len == 2){
                        TCLOCAL_TYPE local_idx = (r % ROW_WINDOW) * COL_WINDOW + c_idx % COL_WINDOW;
                        if (local_idx < 64){
                            TCLOCAL_TYPE local_idx_mask = 1ULL << local_idx;
                            tcLocalBittmp[c_idx / COL_WINDOW * 2] |= local_idx_mask;
                        } else{
                            TCLOCAL_TYPE local_idx_mask = 1ULL << (local_idx-64);
                            tcLocalBittmp[c_idx / COL_WINDOW * 2 + 1] |= local_idx_mask;
                        }
                    } else {
                        std::cout << "ERROE: in CSR2AdpBME(): TC shape is not surportable!!" << std::endl;
                    }
                    datatmp[c_idx / COL_WINDOW].push_back(csr.data[nnz_id]);
                }
            }
            tcLocalBit.insert(tcLocalBit.end(), tcLocalBittmp.begin(), tcLocalBittmp.end());

            for (vint i = 0; i < datatmp.size(); ++i) {
                data.insert(data.end(), datatmp[i].begin(), datatmp[i].end());
            }
            total_tc_num += window_tc_num;
        }

        if(current_group_size != 0 ){
            groupOffset.push_back(current_group_size);
        }
        for(vint i = 0; i < groupOffset.size()-1; ++i){
            groupOffset[i+1] += groupOffset[i];
        }    

        tcOffset.insert(tcOffset.begin(), 0);
        for(vint i = 0; i < tcOffset.size()-1; ++i){
            tcOffset[i+1] += tcOffset[i];
        }
    }

    void show() const override{
        std::cout << "\n========== AdpBME =========:\n";

        std::cout << "groupOffset:" << std::endl;
        for(auto iter = groupOffset.begin(); iter != groupOffset.end(); ++iter){
            std::cout << *iter << " ";
        }
        std::cout << std::endl;

        std::cout << "tcOffset:" << std::endl;
        for(auto iter = tcOffset.begin(); iter != tcOffset.end(); ++iter){
            std::cout << *iter << " ";
        }
        std::cout << std::endl;

        std::cout << "rowIndices:" << std::endl;
        for(auto iter = rowIndices.begin(); iter != rowIndices.end(); ++iter){
            std::cout << *iter << " ";
        }
        std::cout << std::endl;

        std::cout << "tcLocalBit:" << std::endl;
        // vint cnt = 0;
        std::cout << "tcLocalBit.size() = " << tcLocalBit.size() << std::endl;
        for(auto iter = tcLocalBit.begin(); iter != tcLocalBit.end(); ++iter){
            std::cout << std::bitset<64>(*iter) << std::endl;
        }
        std::cout << std::endl;

        std::cout << "sparseA2B:" << std::endl;
        vint cnt = 0;
        for (auto iter = sparseA2B.begin(); iter != sparseA2B.end(); ++iter) {
            std::cout << *iter << " ";
            ++cnt;
            if (cnt == COL_WINDOW) {
                std::cout << std::endl; 
                cnt = 0;
            }
        }

        std::cout << std::endl;
        std::cout << "data:" << std::endl;

        for (vint i = 0; i < groupOffset.size()-1; ++i) {
            for(vint nnz_id = groupOffset[i]; nnz_id < groupOffset[i+1]; ++nnz_id) {
                std::cout << data[nnz_id] << " ";
            }
            std::cout << std::endl;
        }
    }

    long long printSize(bool print = true) const override {
        long long s1 = groupOffset.size() * sizeof(int);
        long long s2 = tcOffset.size() * sizeof(int);
        long long s3 = rowIndices.size() * sizeof(int); 
        long long s4 = tcLocalBit.size() * sizeof(TCLOCAL_TYPE);
        long long s5 = sparseA2B.size() * sizeof(int);
        long long total = s1 + s2 + s3 + s4 + s5;

        if (print) {
            std::cout << "\n========== size of AdpBME (ours) =========\n";
            std::cout << "total size(without data array): " << total << " bytes" << std::endl;
            std::cout << "groupOffset: " << s1 << " bytes" << std::endl;
            std::cout << "tcOffset: " << s2 << " bytes" << std::endl;
            std::cout << "rowIndices: " << s3 << " bytes"  << std::endl;
            std::cout << "tcLocalBit: " << s4 << " bytes"  << std::endl;
            std::cout << "sparseA2B: " << s5 << " bytes"  << std::endl;            
            std::cout << std::endl;
        }

        return total;
    }
};
