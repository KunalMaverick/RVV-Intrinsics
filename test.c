#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <riscv_vector.h>
#include "util.h"

int main() {
    srand(time(NULL));
    clock_t start, end;
    double time_taken;
    size_t n = 65536;

    u_int64_t* B = allocate_tensor_1d_uint64(n);
    u_int64_t* C = allocate_tensor_1d_uint64(n);
    u_int64_t* A = allocate_tensor_1d_uint64(n);

    fill_u64_tensor_rvv_rand(A, n);
    fill_u64_tensor_rvv_rand(B, n);

    printf("Vector-Vector Addition:\n");
    start = clock();
    add_vv_u64(n, A, B, C);
    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC; // Convert to seconds
    printf("Vector-Vector Addition took %f seconds to execute \n", time_taken);
    // for (size_t i = 0; i < n; i++) {
    //     printf("%d + %d = %d\n", A[i], B[i], C[i]);
    // }

    printf("Vector-Vector Subtraction:\n");
    start = clock();
    sub_vv_u64(n, A, B, C);
    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC; // Convert to seconds
    printf("Vector-Vector Subtraction took %f seconds to execute \n", time_taken);

    printf("Vector-Scalar Addition (scalar = 10):\n");
    start = clock();
    add_vx_u64(n, A, 10, C);
    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC; // Convert to seconds
    printf("Vector-Scalr Addition took %f seconds to execute \n", time_taken);
    // for (size_t i = 0; i < n; i++) {
    //     printf("%d + 10 = %d\n", A[i], C[i]);
    // }

    printf("Vector-Scalar Subtraction (scalar = 10):\n");
    start = clock();
    sub_vx_u64(n, A, 10, C);
    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC; // Convert to seconds
    printf("Vector-Scalar Subtraction took %f seconds to execute \n", time_taken);

    printf("Vector-Vector Addition W/O RVV:\n");
    start = clock();
    for(int i = 0; i<n ; i++)
    {
        C[i] = A[i] + B[i];
    }
    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC; // Convert to seconds
    printf("Vector-Vector Addition took %f seconds to execute \n", time_taken);
    
    printf("Vector-Vector Subtraction W/O RVV:\n");
    start = clock();
    for(int i = 0; i<n ; i++)
    {
        C[i] = A[i] - B[i];
    }
    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC; // Convert to seconds
    printf("Vector-Vector Subtraction took %f seconds to execute \n", time_taken);
    
    printf("Vector-Scalar Subtraction W/O RVV:\n");
    start = clock();
    for(int i = 0; i<n ; i++)
    {
        C[i] = A[i] - 10;
    }
    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC; // Convert to seconds
    printf("Vector-Scalar Subtraction took %f seconds to execute \n", time_taken);
    

    free_tensor_safe((void**)&A);
    free_tensor_safe((void**)&B);
    free_tensor_safe((void**)&C);

    return 0;
}


// int main() {

//     srand(time(NULL));
//     // Test sizes, including edge cases
//     size_t test_sizes[] = {0, 1, 8, 16, 31, 32, 33, 64, 100};
//     size_t num_tests = sizeof(test_sizes)/sizeof(test_sizes[0]);

//     for (size_t t = 0; t < num_tests; t++) {
//         size_t n = test_sizes[t];
//         printf("\n=== Testing tensor size: %zu ===\n", n);

//         // ---- float ----
//         float* f = allocate_tensor_1d_float(n);
//         if (f) {
//             fill_float_tensor_rvv_rand(f, n);
//             printf("float tensor: ");
//             for (size_t i = 0; i < n; i++) printf("%f ", f[i]);
//             printf("\n");
//             free_tensor_safe((void**)&f);
//             printf("After free: %p\n", (void*)f);
//         }

//         // ---- int ----
//         int* i_tensor = allocate_tensor_1d_int(n);
//         if (i_tensor) {
//             fill_int_tensor_rvv_rand(i_tensor, n);
//             printf("int tensor: ");
//             for (size_t i = 0; i < n; i++) printf("%d ", i_tensor[i]);
//             printf("\n");
//             free_tensor_safe((void**)&i_tensor);
//             printf("After free: %p\n", (void*)i_tensor);
//         }

//         // ---- uint32_t ----
//         uint32_t* u32 = allocate_tensor_1d_uint32(n);
//         if (u32) {
//             fill_u32_tensor_rvv_rand(u32, n);
//             printf("uint32 tensor: ");
//             for (size_t i = 0; i < n; i++) printf("%u ", u32[i]);
//             printf("\n");
//             free_tensor_safe((void**)&u32);
//             printf("After free: %p\n", (void*)u32);
//         }

//         // ---- uint64_t ----
//         uint64_t* u64 = allocate_tensor_1d_uint64(n);
//         if (u64) {
//             fill_u64_tensor_rvv_rand(u64, n);
//             printf("uint64 tensor: ");
//             for (size_t i = 0; i < n; i++) printf("%llu ", (unsigned long long)u64[i]);
//             printf("\n");
//             free_tensor_safe((void**)&u64);
//             printf("After free: %p\n", (void*)u64);
//         }
//     }

//     printf("\n=== All tests completed ===\n");
//     return 0;
// }




// int main() {
//     size_t n = 10;

//     float* f = allocate_tensor_1d_float(n);
//     double* d = allocate_tensor_1d_double(n);
//     int* i = allocate_tensor_1d_int(n);
//     uint64_t* u64 = allocate_tensor_1d_uint64(n);

//     if (!f || !d || !i || !u64) return 1;

//     for (size_t j = 0; j < n; j++) {
//         f[j] = (float)j;
//         d[j] = (double)j * 2.5;
//         i[j] = (int)(j * 3);
//         u64[j] = (uint64_t)j * 100;
//     }

//     printf("Tensors:\n");
//     for (size_t j = 0; j < n; j++) {
//         printf("[%zu] f=%f, d=%lf, i=%d, u64=%llu\n",
//                j, f[j], d[j], i[j], (unsigned long long)u64[j]);
//     }

//     printf("\n--- Freeing tensors ---\n");

//     printf("Freeing float tensor...\n");
//     free_tensor_1d(f);
//     printf("Float tensor freed.\n");

//     printf("Freeing double tensor...\n");
//     free_tensor_1d(d);
//     printf("Double tensor freed.\n");

//     printf("Freeing int tensor...\n");
//     free_tensor_1d(i);
//     printf("Int tensor freed.\n");

//     printf("Freeing uint64 tensor...\n");
//     free_tensor_1d(u64);
//     printf("Uint64 tensor freed.\n");

//     printf("--- All tensors freed successfully ---\n");

//      printf("Tensors:\n");
//     for (size_t j = 0; j < n; j++) {
//         printf("[%zu] f=%f, d=%lf, i=%d, u64=%llu\n",
//                j, f[j], d[j], i[j], (unsigned long long)u64[j]);
//     }

//     return 0;
// }

