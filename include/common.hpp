#pragma once
#include <omp.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cusparse.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <deque>

#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <tuple>
#include <algorithm>
#include <memory>
#include <iomanip>
#include <numeric>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <bitset>


// #ifdef  tf
#define MAT_VAL_TYPE float        // for tf32 computation, precision::tf32 -> float
// #else
// #define MAT_VAL_TYPE double
// #endif

#define MAT_PTR_TYPE uint32_t

#define MMA_M   16
#define MMA_K   8
#define MMA_N   8

#define ROW_WINDOW      8

#define COL_WINDOW      8
#define ROW_WINDOW_R    8
#define COL_WINDOW_R    16
#define BASE_NNZ        16
#define MAX_LEVELS      1000
#define FEATURE_DIM     512
#define GROUP_LEN       32
#define TARGET_NUM      16

#define BIT_TYPE        bool
#define TCLOCAL_TYPE    uint64_t
#define WARP_SIZE       32
#define THREADS_PER_BLK 1024
#define WARP_NUM    (THREADS_PER_BLK / WARP_SIZE)
#define IMBALANCE_THRESHOLD 8

typedef uint32_t vint;
typedef uint4    row_int;

#define BLK_H 16 
#define BLK_W 8

/*************************** Return Error codes ******************************/
#define ERROR_NOT_MATCH         1
#define ERROR_OPEN_FILE         2
#define ERROR_READ_MTX          3
#define ERROR_ARGS              4

/***********************/
#define MM_MAX_LINE_LENGTH   1025
#define MatrixMarketBanner   "%%MatrixMarket"
#define MM_MAX_TOKEN_LENGTH  64
/***********************/

/*************************** Matrix Load codes ******************************/
#define SUCCESS                  0
#define MM_LOADING_FILE_ERROR   -1
#define MM_READ_BANNER_ERROR    -2
#define RET_ERROR               -4

/*********************** Matrix Market error codes *************************/
#define MM_COULD_NOT_READ_FILE	11
#define MM_PREMATURE_EOF		12
#define MM_NOT_MTX				13
#define MM_NO_HEADER			14
#define MM_UNSUPPORTED_TYPE		15
#define MM_LINE_TOO_LONG		16
#define MM_COULD_NOT_WRITE_FILE	17

/*************************** Matrix String *******************************/
#define MM_MTX_STR		    "matrix"
#define MM_DENSE_STR	    "array"
#define MM_SPARSE_STR	    "coordinate"
#define MM_COMPLEX_STR	    "complex"
#define MM_REAL_STR		    "real"
#define MM_INT_STR		    "integer"
#define MM_GENERAL_STR      "general"
#define MM_SYMM_STR		    "symmetric"
#define MM_HERM_STR		    "Hermitian"
#define MM_SKEW_STR		    "skew-symmetric"
#define MM_PATTERN_STR      "pattern"


/********************* MM_typecode query functions ***************************/
#define mm_is_matrix(typecode)	    ((typecode)[0]=='M')
#define mm_is_sparse(typecode)	    ((typecode)[1]=='C')
#define mm_is_dense(typecode)	    ((typecode)[1]=='A')
#define mm_is_array(typecode)	    ((typecode)[1]=='A')
#define mm_is_complex(typecode)	    ((typecode)[2]=='C')
#define mm_is_real(typecode)		((typecode)[2]=='R')
#define mm_is_pattern(typecode) 	((typecode)[2]=='P')
#define mm_is_integer(typecode)     ((typecode)[2]=='I')
#define mm_is_symmetric(typecode)   ((typecode)[3]=='S')
#define mm_is_general(typecode)	    ((typecode)[3]=='G')
#define mm_is_skew(typecode)	    ((typecode)[3]=='K')
#define mm_is_hermitian(typecode)   ((typecode)[3]=='H')

/********************* MM_typecode modify functions ***************************/
#define mm_set_matrix(typecode)	        ((*typecode)[0]='M')
#define mm_set_coordinate(typecode)	    ((*typecode)[1]='C')
#define mm_set_array(typecode)	        ((*typecode)[1]='A')
#define mm_set_dense(typecode)	        mm_set_array(typecode)
#define mm_set_sparse(typecode)	        mm_set_coordinate(typecode)
#define mm_set_complex(typecode)        ((*typecode)[2]='C')
#define mm_set_real(typecode)	        ((*typecode)[2]='R')
#define mm_set_pattern(typecode)        ((*typecode)[2]='P')
#define mm_set_integer(typecode)        ((*typecode)[2]='I')
#define mm_set_symmetric(typecode)      ((*typecode)[3]='S')
#define mm_set_general(typecode)        ((*typecode)[3]='G')
#define mm_set_skew(typecode)	        ((*typecode)[3]='K')
#define mm_set_hermitian(typecode)      ((*typecode)[3]='H')
#define mm_clear_typecode(typecode)     ((*typecode)[0]=(*typecode)[1]=(*typecode)[2]=' ',(*typecode)[3]='G')
#define mm_initialize_typecode(typecode) mm_clear_typecode(typecode)
