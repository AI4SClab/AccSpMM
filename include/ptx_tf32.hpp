/**
 * @file ptx_tf32.hpp
 * @author Haisha Zhao
 * @date 2025-04-02
 * 
 * @copyright MIT License (c) 2025 Haisha Zhao
*/

#pragma once
#include "utils.hpp"

__device__ __forceinline__
void tf32_m16n8k8(MAT_VAL_TYPE* MatA, MAT_VAL_TYPE* MatB, MAT_VAL_TYPE* MatC) {
    vint const* A   = reinterpret_cast<vint const*>(MatA);
    vint const* B   = reinterpret_cast<vint const*>(MatB);
    float* C        = reinterpret_cast<float*>(MatC);

    asm volatile(
        "cvt.rna.tf32.f32 %4, %4;\n"
        "cvt.rna.tf32.f32 %5, %5;\n"
        "cvt.rna.tf32.f32 %6, %6;\n"
        "cvt.rna.tf32.f32 %7, %7;\n"
        "cvt.rna.tf32.f32 %8, %8;\n"
        "cvt.rna.tf32.f32 %9, %9;\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%0, %1, %2, %3};\n"
        :"+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])      // output
        :"r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
         "r"(B[0]), "r"(B[1])
    );
}

__device__ __forceinline__
void tf32_m16n8k8_detail(
    MAT_VAL_TYPE MatA0, 
    MAT_VAL_TYPE MatA1, 
    MAT_VAL_TYPE MatA2, 
    MAT_VAL_TYPE MatA3, 
    MAT_VAL_TYPE MatB0,
    MAT_VAL_TYPE MatB1, 
    MAT_VAL_TYPE C0,
    MAT_VAL_TYPE C1,
    MAT_VAL_TYPE C2,
    MAT_VAL_TYPE C3
) {
    vint const A0   = MatA0;
    vint const A1   = MatA1;
    vint const A2   = MatA2;
    vint const A3   = MatA3;
    vint const B0   = MatB0;
    vint const B1   = MatB1;

    asm volatile(
        "cvt.rna.tf32.f32 %4, %4;\n"
        "cvt.rna.tf32.f32 %5, %5;\n"
        "cvt.rna.tf32.f32 %6, %6;\n"
        "cvt.rna.tf32.f32 %7, %7;\n"
        "cvt.rna.tf32.f32 %8, %8;\n"
        "cvt.rna.tf32.f32 %9, %9;\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32"
        "{%0, %1, %2, %3},"
        "{%4, %5, %6, %7},"
        "{%8, %9},"
        "{%0, %1, %2, %3};\n"
        :"+f"(C0), "+f"(C1), "+f"(C2), "+f"(C3)      // output
        :"r"(A0), "r"(A1), "r"(A2), "r"(A3),
         "r"(B0), "r"(B1)
    );
}

__device__ __forceinline__
void tf32_m16n8k4(MAT_VAL_TYPE* MatA, MAT_VAL_TYPE* MatB, MAT_VAL_TYPE* MatC) {
    vint const* A   = reinterpret_cast<vint const*>(MatA);
    vint const* B   = reinterpret_cast<vint const*>(MatB);
    float *C        = reinterpret_cast<float*>(MatC);

    asm volatile(
        "cvt.rna.tf32.f32 %4, %4;\n"
        "cvt.rna.tf32.f32 %5, %5;\n"
        "cvt.rna.tf32.f32 %6, %6;\n"
        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32"
        "{%0, %1, %2, %3},"
        "{%4, %5},"
        "{%6},"
        "{%0, %1, %2, %3};\n"
        :"+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])      // output
        :"r"(A[0]), "r"(A[1]),
         "r"(B[0])
    );
}

__device__ __forceinline__
void wait_group() {
    asm volatile(
        "cp.async.commit_group;\n"
        "cp.async.wait_group 0;\n"
        ::
    );
}

__device__ __forceinline__
void async_copy (MAT_PTR_TYPE shared_addr, const MAT_VAL_TYPE* val) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shared_addr), "l"(val));
}

__device__ __forceinline__
void async_copy_idx (MAT_PTR_TYPE shared_addr, const vint* val) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(shared_addr), "l"(val));
}

// Cache in L1, L2.
__device__ __forceinline__
MAT_VAL_TYPE load_fp32_from_global(const MAT_VAL_TYPE* a) {
    MAT_VAL_TYPE r;
    asm volatile("ld.global.ca.f32 %0, [%1];" : "=f"(r) : "l"(a));
    return r;
}

__device__ __forceinline__
MAT_VAL_TYPE load_fp32_from_global_cs(const MAT_VAL_TYPE* a) {
    MAT_VAL_TYPE r;
    asm volatile("ld.global.cs.f32 %0, [%1];" : "=f"(r) : "l"(a));
    return r;
}

// Don't cache and fetch again.
__device__ __forceinline__
MAT_VAL_TYPE load_fp32_from_global2shared(const MAT_VAL_TYPE* a) {
    MAT_VAL_TYPE r;
    asm volatile("ld.global.cv.f32 %0, [%1];" : "=f"(r) : "l"(a));
    return r;
}

__device__ __forceinline__ 
vint load_int_from_global(const vint* a) {
    int r;
    asm volatile("ld.global.ca.s32 %0, [%1];" : "=r"(r) : "l"(a));      // s32:signed integer
    return r;
}

// recently least used cache friendly
__device__ __forceinline__
void store_fp32_to_global(MAT_VAL_TYPE* a, MAT_VAL_TYPE v) {
    asm volatile("st.global.wt.f32 [%0], %1;" :: "l"(a), "f"(v));
}

__device__ __forceinline__
MAT_VAL_TYPE load_fp32_from_shared1(const MAT_PTR_TYPE a) {
    MAT_VAL_TYPE r;
    asm volatile("ld.shared.cs.f32 %0, [%1];" : "=f"(r) : "r"(a));
    return r;
}

__device__ __forceinline__
float4 vector_fetch_fp32V4(const float4 *ptr) {
    float4 ret;
    asm volatile (
        "ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
        : "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
        : "l"(ptr)
    );
    return ret;
}

__device__ __forceinline__
float2 vector_fetch_fp32V2(const float2 *ptr) {
    float2 ret;
    asm volatile (
        "ld.global.v2.f32 {%0, %1}, [%2];"
        : "=f"(ret.x), "=f"(ret.y)
        : "l"(ptr)
    );
    return ret;
}

__device__ __forceinline__
MAT_VAL_TYPE load_int_from_shared(const MAT_PTR_TYPE a) {
    vint r;
    asm volatile("ld.shared.cs.s32 %0, [%1];" : "=r"(r) : "r"(a));
    return r;
}

__device__ __forceinline__
float2 ld_shared_float2(uint a) {
    float2 v;
    asm volatile ("ld.shared.v2.f32 {%0, %1}, [%2];"  : "=f"(v.x),"=f"(v.y) : "r"(a*4));
    return v;
}

__device__ __forceinline__
float4 ld_shared_float4(uint a) {
    float4 v;
    asm volatile ("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"  : "=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w) : "r"(a*4));
    return v;
}
