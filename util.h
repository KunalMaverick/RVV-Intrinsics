// util.h
// common utilities for the test code under exmaples/

#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <riscv_vector.h>


// ---- float ----
static inline float* allocate_tensor_1d_float(size_t n) {
    float* tensor = (float*)malloc(n * sizeof(float));
    if (tensor == NULL) {
        fprintf(stderr, "[alloc_float] Failed to allocate tensor of size %zu\n", n);
        return NULL;
    }
    return tensor;
}


// ---- int ----
static inline int* allocate_tensor_1d_int(size_t n) {
    int* tensor = (int*)malloc(n * sizeof(int));
    if (tensor == NULL) {
        fprintf(stderr, "[alloc_int] Failed to allocate tensor of size %zu\n", n);
        return NULL;
    }
    return tensor;
}

// ---- uint32_t ----
static inline uint32_t* allocate_tensor_1d_uint32(size_t n) {
    uint32_t* tensor = (uint32_t*)malloc(n * sizeof(uint32_t));
    if (tensor == NULL) {
        fprintf(stderr, "[alloc_uint32] Failed to allocate tensor of size %zu\n", n);
        return NULL;
    }
    return tensor;
}

// ---- uint64_t ----
static inline uint64_t* allocate_tensor_1d_uint64(size_t n) {
    uint64_t* tensor = (uint64_t*)malloc(n * sizeof(uint64_t));
    if (tensor == NULL) {
        fprintf(stderr, "[alloc_uint64] Failed to allocate tensor of size %zu\n", n);
        return NULL;
    }
    return tensor;
}


// Example usage with RVV intrinsics
static inline void fill_float_tensor_rvv_rand(float *tensor, size_t n) {
    
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        float tmp[vl];
        for (size_t j = 0; j < vl; j++) {
            tmp[j] = (float)rand();
        }
        vfloat32m8_t vdata = __riscv_vle32_v_f32m8(tmp, vl);
        __riscv_vse32_v_f32m8(tensor + i, vdata, vl);
        i += vl;
    }
}

static inline void fill_int_tensor_rvv_rand(int *tensor, size_t n) {
    
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        int tmp[vl];
        for (size_t j = 0; j < vl; j++) {
            tmp[j] = (int)rand();
        }
        vint32m8_t vdata = __riscv_vle32_v_i32m8(tmp, vl);
        __riscv_vse32_v_i32m8(tensor + i, vdata, vl);
        i += vl;
    }
}



static inline void fill_u32_tensor_rvv_rand(uint32_t *tensor, size_t n) {
    
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e32m8(n - i);
        uint32_t tmp[vl];
        for (size_t j = 0; j < vl; j++) {
            tmp[j] = (uint32_t)rand();
        }
        vuint32m8_t vdata = __riscv_vle32_v_u32m8(tmp, vl);
        __riscv_vse32_v_u32m8(tensor + i, vdata, vl);
        i += vl;
    }
}


static inline void fill_u64_tensor_rvv_rand(uint64_t *tensor, size_t n) {
    
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e64m8(n - i);
        uint64_t tmp[vl];
        for (size_t j = 0; j < vl; j++) {
            tmp[j] = (uint64_t)rand();
        }
        vuint64m8_t vdata = __riscv_vle64_v_u64m8(tmp, vl);
        __riscv_vse64_v_u64m8(tensor + i, vdata, vl);
        i += vl;
    }
}

// ---- free function ----
static inline void free_tensor_safe(void** ptr) {
    if (ptr && *ptr) {
        printf("[DEBUG] Freeing tensor at address %p\n", *ptr);
        free(*ptr);
        *ptr = NULL;  // nullify pointer to avoid dangling access
        printf("[DEBUG] Tensor freed and pointer set to NULL\n");
    }
}

// -------------------- Vector operations --------------------
void add_vv_int32(size_t n, const int32_t *a, const int32_t *b, int32_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vint32m8_t va = __riscv_vle32_v_i32m8(a, vl);
        vint32m8_t vb = __riscv_vle32_v_i32m8(b, vl);
        vint32m8_t vc = __riscv_vadd_vv_i32m8(va, vb, vl);
        __riscv_vse32_v_i32m8(c, vc, vl);
    }
}

void add_vx_int32(size_t n, const int32_t *a, int32_t b, int32_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, c += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vint32m8_t va = __riscv_vle32_v_i32m8(a, vl);
        vint32m8_t vc = __riscv_vadd_vx_i32m8(va, b, vl);
        __riscv_vse32_v_i32m8(c, vc, vl);
    }
}

void sub_vv_int32(size_t n, const int32_t *a, const int32_t *b, int32_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vint32m8_t va = __riscv_vle32_v_i32m8(a, vl);
        vint32m8_t vb = __riscv_vle32_v_i32m8(b, vl);
        vint32m8_t vc = __riscv_vsub_vv_i32m8(va, vb, vl);
        __riscv_vse32_v_i32m8(c, vc, vl);
    }
}

void sub_vx_int32(size_t n, const int32_t *a, int32_t b, int32_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, c += vl) {
        vl = __riscv_vsetvl_e32m8(n);
        vint32m8_t va = __riscv_vle32_v_i32m8(a, vl);
        vint32m8_t vc = __riscv_vsub_vx_i32m8(va, b, vl);
        __riscv_vse32_v_i32m8(c, vc, vl);
    }
}

void add_vv_u64(size_t n, const uint64_t *a, const uint64_t *b, uint64_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e64m8(n);              // 64-bit elements
        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl); // load a
        vuint64m8_t vb = __riscv_vle64_v_u64m8(b, vl); // load b
        vuint64m8_t vc = __riscv_vadd_vv_u64m8(va, vb, vl); // vector-vector addition
        __riscv_vse64_v_u64m8(c, vc, vl);          // store result in c
    }
}

void add_vx_u64(size_t n, const uint64_t *a, uint64_t b, uint64_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, c += vl) {
        vl = __riscv_vsetvl_e64m8(n);                 // 64-bit elements
        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl); // load vector a
        vuint64m8_t vc = __riscv_vadd_vx_u64m8(va, b, vl); // add scalar b
        __riscv_vse64_v_u64m8(c, vc, vl);             // store result
    }
}

void sub_vv_u64(size_t n, const uint64_t *a, const uint64_t *b, uint64_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e64m8(n);              // 64-bit elements
        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl); // load a
        vuint64m8_t vb = __riscv_vle64_v_u64m8(b, vl); // load b
        vuint64m8_t vc = __riscv_vsub_vv_u64m8(va, vb, vl); // vector-vector subtraction
        __riscv_vse64_v_u64m8(c, vc, vl);          // store result in c
    }
}

void sub_vx_u64(size_t n, const uint64_t *a, uint64_t b, uint64_t *c) {
    size_t vl;
    for (; n > 0; n -= vl, a += vl, c += vl) {
        vl = __riscv_vsetvl_e64m8(n);
        vuint64m8_t va = __riscv_vle64_v_u64m8(a, vl); // load vector a
        vuint64m8_t vc = __riscv_vsub_vx_u64m8(va, b, vl); // subtract scalar b
        __riscv_vse64_v_u64m8(c, vc, vl);               // store result
    }
}


// void gen_rand_1d(double *a, int n) {
//   for (int i = 0; i < n; ++i)
//     a[i] = (double)rand() / (double)RAND_MAX + (double)(rand() % 1000);
// }

// void gen_string(char *s, int n) {
//   // char value range: -128 ~ 127
//   for (int i = 0; i < n - 1; ++i)
//     s[i] = (char)(rand() % 95) + 32;  
//   s[n - 1] = '\0';
// }

// void gen_rand_2d(double **ar, int n, int m) {
//   for (int i = 0; i < n; ++i)
//     for (int j = 0; j < m; ++j)
//       ar[i][j] = (double)rand() / (double)RAND_MAX + (double)(rand() % 1000);
// }

// void print_string(const char *a, const char *name) {
//   printf("const char *%s = \"", name);
//   int i = 0;
//   while (a[i] != 0)
//     putchar(a[i++]);
//   printf("\"\n");
//   puts("");
// }

// void print_array_1d(double *a, int n, const char *type, const char *name) {
//   printf("%s %s[%d] = {\n", type, name, n);
//   for (int i = 0; i < n; ++i) {
//     printf("%06.2f%s", a[i], i != n - 1 ? "," : "};\n");
//     if (i % 10 == 9)
//       puts("");
//   }
//   puts("");
// }

// void print_array_2d(double **a, int n, int m, const char *type,
//                     const char *name) {
//   printf("%s %s[%d][%d] = {\n", type, name, n, m);
//   for (int i = 0; i < n; ++i) {
//     for (int j = 0; j < m; ++j) {
//       printf("%06.2f", a[i][j]);
//       if (j == m - 1)
//         puts(i == n - 1 ? "};" : ",");
//       else
//         putchar(',');
//     }
//   }
//   puts("");
// }

// bool double_eq(double golden, double actual, double relErr) {
//   return (fabs(actual - golden) < relErr);
// }

// bool compare_1d(double *golden, double *actual, int n) {
//   for (int i = 0; i < n; ++i)
//     if (!double_eq(golden[i], actual[i], 1e-6))
//       return false;
//   return true;
// }

// bool compare_string(const char *golden, const char *actual, int n) {
//   for (int i = 0; i < n; ++i)
//     if (golden[i] != actual[i])
//       return false;
//   return true;
// }

// bool compare_2d(double **golden, double **actual, int n, int m) {
//   for (int i = 0; i < n; ++i)
//     for (int j = 0; j < m; ++j)
//       if (!double_eq(golden[i][j], actual[i][j], 1e-6))
//         return false;
//   return true;
// }

// double **alloc_array_2d(int n, int m) {
//   double **ret;
//   ret = (double **)malloc(sizeof(double *) * n);
//   for (int i = 0; i < n; ++i)
//     ret[i] = (double *)malloc(sizeof(double) * m);
//   return ret;
// }

// void init_array_one_1d(double *ar, int n) {
//   for (int i = 0; i < n; ++i)
//     ar[i] = 1;
// }

// void init_array_one_2d(double **ar, int n, int m) {
//   for (int i = 0; i < n; ++i)
//     for (int j = 0; j < m; ++j)
//       ar[i][j] = 1;
// }