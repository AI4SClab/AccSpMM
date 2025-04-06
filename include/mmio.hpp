/* 
*   Matrix Market I/O library for ANSI C
*
*   See http://math.nist.gov/MatrixMarket for details.
*
*/

#pragma once
#include <string>
#include <cstring>
#include "format.hpp"

typedef char MM_typecode[4];
int mm_read_banner(FILE* file, MM_typecode* matcode);
int mm_read_mtx_crd_size(FILE* file, vint* M, vint* N, vint* nz);
template<class DataType>
int read_from_mtx(const char* filename, COO<DataType>* coo);

template<class DataType>
void handle_coo(FILE* file, vint m, vint n, vint nnz, bool isInteger, bool isReal, bool isPattern, bool isSymmetric, bool isComplex, COO<DataType>* coo);
bool mm_is_valid(MM_typecode matcode);		/* too complex for a macro */

/* ------------------------ */
int mm_read_banner(FILE* file, MM_typecode* matcode) {
    // TODO
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH];
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    // char* p;

    mm_clear_typecode(matcode);

    if(fgets(line, MM_MAX_LINE_LENGTH, file) == NULL) return MM_PREMATURE_EOF;
    if(sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, storage_scheme) != 5) return MM_PREMATURE_EOF;
    
    // check whether a banner exists.
    if(strcmp(banner, MatrixMarketBanner) != 0) return MM_NO_HEADER;

    // convert to lower case
    // for (p=mtx; *p!='\0'; *p=tolower(*p), ++p);  
    // for (p=crd; *p!='\0'; *p=tolower(*p), ++p);
    // for (p=data_type; *p!='\0'; *p=tolower(*p), ++p);
    // for (p=storage_scheme; *p!='\0'; *p=tolower(*p), ++p);

    // check whether is matrix
    if(strcmp(mtx, MM_MTX_STR) != 0) return MM_UNSUPPORTED_TYPE;

    mm_set_matrix(matcode);

    // check crd field
    if(strcmp(crd, MM_SPARSE_STR) == 0)     mm_set_sparse(matcode);
    else if(strcmp(crd, MM_DENSE_STR) == 0) mm_set_dense(matcode);

    // check data_type field
    if(strcmp(data_type, MM_REAL_STR) == 0)         mm_set_real(matcode);
    else if(strcmp(data_type, MM_COMPLEX_STR) == 0) mm_set_complex(matcode);
    else if(strcmp(data_type, MM_PATTERN_STR) == 0) mm_set_pattern(matcode);
    else if(strcmp(data_type, MM_INT_STR) == 0)     mm_set_integer(matcode);
    else return MM_UNSUPPORTED_TYPE;

    // check storage_scheme field
    if(strcmp(storage_scheme, MM_GENERAL_STR) == 0)     mm_set_general(matcode);
    else if(strcmp(storage_scheme, MM_SYMM_STR) == 0)   mm_set_symmetric(matcode);
    else if(strcmp(storage_scheme, MM_HERM_STR) == 0)   mm_set_hermitian(matcode);
    else if(strcmp(storage_scheme, MM_SKEW_STR) == 0)   mm_set_skew(matcode);
    else return MM_UNSUPPORTED_TYPE;

    return SUCCESS;
}

int mm_read_mtx_crd_size(FILE* file, vint* M, vint* N, vint* nz) {
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;

    // initialize;
    *M = *N = *nz = 0;

    do {
        if(fgets(line, MM_MAX_LINE_LENGTH, file) == NULL) return MM_PREMATURE_EOF;
    } while(line[0] == '%');

    if(sscanf(line, "%u %u %u", M, N, nz) == 3) return SUCCESS;
    else do {
        num_items_read = fscanf(file, "%u %u %u", M, N, nz);
        if(num_items_read == EOF) return MM_PREMATURE_EOF;
        
    } while(num_items_read != 3);

    return SUCCESS;
}


bool mm_is_valid(MM_typecode matcode) {
    if(!mm_is_matrix(matcode)) return false;
    if(mm_is_dense(matcode) && mm_is_pattern(matcode)) return true;
    if(mm_is_real(matcode) && mm_is_hermitian(matcode)) return true;
    if((mm_is_pattern(matcode) && mm_is_hermitian(matcode)) || (mm_is_skew(matcode))) return true;

    return false;
}

// store naive .mtx data to coo format
template <class DataType>
void handle_coo(FILE* file, vint m, vint n, vint nnz, bool isInteger, bool isReal, bool isPattern, bool isSymmetric, bool isComplex, COO<DataType>* coo) {
    coo->rows = m; coo->cols = n; coo->nnz = nnz;
    vint diag_cnt = 0;

    vint row_id = 0, col_id = 0;
    vint* tmp_row = (vint*)malloc(sizeof(vint) * nnz);
    vint* tmp_col = (vint*)malloc(sizeof(vint) * nnz);
    DataType* tmp_data = (DataType*)malloc(sizeof(DataType) * nnz);
    double data_f, data_im;

    // get diag_cnt;
    struct timeval t1, t2;
    // gettimeofday(&t1, NULL);
    // #pragma omp parallel for ordered
    for(vint i = 0; i < nnz; ++i) {
        if(isReal)              fscanf(file, "%u %u %lg\n", &row_id, &col_id, &(data_f));
        else if(isComplex)      fscanf(file, "%u %u %lg %lg\n", &row_id, &col_id, &data_f, &data_im);
        else if(isInteger)      fscanf(file, "%u %u %d\n", &row_id, &col_id, &data_f);
        else if(isPattern)  {   fscanf(file, "%u %u\n", &row_id, &col_id); data_f = 1.0;  }
        tmp_row[i] = row_id-1; tmp_col[i] = col_id-1; tmp_data[i] = data_f;
        if(row_id == col_id) ++diag_cnt; 
    }
    // gettimeofday(&t2, NULL);
    // double cu_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / 1000;
    // printf("cusparse:%8.4lf ms\n", cu_time);
 
    if(isSymmetric) {
        vint total_nnz = nnz * 2 - diag_cnt;
        coo->nnz = total_nnz;
        coo->row = (vint*)malloc(sizeof(vint) * total_nnz);
        coo->col = (vint*)malloc(sizeof(vint) * total_nnz);
        coo->data = (DataType*)malloc(sizeof(DataType) * total_nnz);

        vint mat_id = 0;
        gettimeofday(&t1, NULL);
        // #pragma omp parallel for
        for(vint i = 0; i < nnz; ++i) {
            coo->row[mat_id] = tmp_row[i]; 
            coo->col[mat_id] = tmp_col[i]; 
            coo->data[mat_id] = tmp_data[i];
            ++mat_id;

            if(tmp_row[i] != tmp_col[i]) {
                coo->row[mat_id] = tmp_col[i]; 
                coo->col[mat_id] = tmp_row[i]; 
                coo->data[mat_id] = tmp_data[i];
                ++mat_id;
            }
        }
        gettimeofday(&t2, NULL);
        double for_time = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0) / 1000;
        printf("for_time:%8.4lf ms\n", for_time);
    } else {  
        coo->row = (vint*)malloc(sizeof(vint) * nnz);
        coo->col = (vint*)malloc(sizeof(vint) * nnz);
        coo->data = (DataType*)malloc(sizeof(DataType) * nnz);
        for(vint i = 0; i < nnz; ++i) {
            coo->row[i] = tmp_row[i]; coo->col[i] = tmp_col[i]; coo->data[i] = tmp_data[i];
        }
    }
    free(tmp_row); free(tmp_col); free(tmp_data);
}

// load data from mtx file
template <class DataType>
int read_from_mtx(const char* filename, COO<DataType>* coo) {
    FILE* file = fopen(filename, "r");
    
    if(file == NULL) {
        perror("Error opening file.");
        return MM_LOADING_FILE_ERROR;
    }

    // read lines
    MM_typecode matcode;
    bool isInteger = false, isReal = false, isPattern = false, isSymmetric_tmp = false, isComplex = false;
    int ret_code;
    vint m_tmp, n_tmp, nnz_tmp;

    int code = mm_read_banner(file, &matcode);
    if(code != SUCCESS) {
        printf("Couldn't process Matrix Market banner. \n");
        return MM_READ_BANNER_ERROR;
    }

    if(mm_is_pattern(matcode))      isPattern = true;
    else if(mm_is_real(matcode))    isReal = true;
    else if(mm_is_complex(matcode)) isComplex = true;
    else if(mm_is_integer(matcode)) isInteger = true;

    ret_code = mm_read_mtx_crd_size(file, &m_tmp, &n_tmp, &nnz_tmp);
    std::cout << "m_tmp: " << m_tmp << ", n_tmp: " << n_tmp << ", nnz_tmp: " << nnz_tmp << std::endl;
    if(ret_code != SUCCESS) return ERROR_READ_MTX;

    if(mm_is_symmetric(matcode) || mm_is_hermitian(matcode)) isSymmetric_tmp = 1;
    else printf("input matrix is not symmetric. \n");

    // -----------------
    handle_coo<DataType>(file, m_tmp, n_tmp, nnz_tmp, isInteger, isReal, isPattern, isSymmetric_tmp, isComplex, coo);

    fclose(file);

    return SUCCESS;
}

