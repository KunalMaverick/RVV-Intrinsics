# RISC-V Vector Extension (RVV) Utilities

A collection of C utilities demonstrating RISC-V Vector (RVV) intrinsics.

## Overview

This project provides:
- **util.h**: Core utility functions for tensor allocation, filling, and vector operations
- **test.c**: Benchmark program comparing RVV-accelerated operations against scalar implementations

## Features

### Tensor Management
- Memory allocation for multiple data types (float, int, uint32_t, uint64_t)
- RVV-optimized random initialization
- Safe memory deallocation with debug output

### Vector Operations
Optimized implementations using RISC-V Vector intrinsics:
- Vector-vector addition        (`add_vv`)
- Vector-scalar addition        (`add_vx`)
- Vector-vector subtraction     (`sub_vv`)
- Vector-scalar subtraction     (`sub_vx`)

Supported types: `int32_t`, `uint64_t`

## Prerequisites

- **Compiler**: RISC-V GCC with vector extension support (`riscv64-unknown-elf-gcc`)
- **Simulator**: Spike RISC-V ISA Simulator with vector extension support
- **ISA**: RV64GCV (RISC-V 64-bit with General, Compressed, and Vector extensions)

## Build Instructions

```bash
riscv64-unknown-elf-gcc -O2 -march=rv64gcv -mabi=lp64d test.c -o test.elf
```

### Compiler Flags
- `-O2`: Optimization level 2
- `-march=rv64gcv`: Target RV64 with General, Compressed, and Vector extensions
- `-mabi=lp64d`: Use LP64D ABI (long and pointers are 64-bit, double in FP registers)

## Running the Program

### On Spike Simulator
```bash
spike --isa=rv64gcv pk test.elf
```

### Save Output to File
```bash
spike --isa=rv64gcv pk test.elf >> test.txt
```

## Benchmark Results

The test program benchmarks operations on 65,536 uint64_t elements:

**With RVV acceleration:**
- Vector-vector addition
- Vector-vector subtraction
- Vector-scalar addition (scalar = 10)
- Vector-scalar subtraction (scalar = 10)

**Without RVV (scalar loops):**
- Equivalent operations for performance comparison

## Code Structure

### util.h
```
├── Allocation functions
│   ├── allocate_tensor_1d_float()
│   ├── allocate_tensor_1d_int()
│   ├── allocate_tensor_1d_uint32()
│   └── allocate_tensor_1d_uint64()
├── Fill functions (RVV-optimized)
│   ├── fill_float_tensor_rvv_rand()
│   ├── fill_int_tensor_rvv_rand()
│   ├── fill_u32_tensor_rvv_rand()
│   └── fill_u64_tensor_rvv_rand()
├── Memory management
│   └── free_tensor_safe()
└── Vector operations (int32 & uint64)
    ├── add_vv_*() / add_vx_*()
    └── sub_vv_*() / sub_vx_*()
```

### test.c
Main benchmark program that:
1. Allocates three uint64_t tensors (A, B, C)
2. Fills A and B with random values
3. Performs timed operations with RVV intrinsics
4. Performs same operations with scalar loops
5. Reports execution times for comparison
6. Deallocates memory

## Key Implementation Details

- **Vector Length Agnostic**: Uses `__riscv_vsetvl_*()` for portable code across different vector lengths
- **LMUL=8**: Operations use `m8` variants for maximum throughput
- **Strip Mining**: Processes arrays in vector-length chunks automatically

<!-- ## Performance Considerations

RVV intrinsics provide significant speedup by:
- Processing multiple elements per instruction
- Utilizing hardware vector units
- Reducing instruction overhead compared to scalar loops

Actual speedup depends on:
- Hardware vector length (VLEN)
- Memory bandwidth
- Data cache behavior -->

## Example Output

```
Vector-Vector Addition:
Vector-Vector Addition took 0.000234 seconds to execute 
Vector-Vector Subtraction:
Vector-Vector Subtraction took 0.000198 seconds to execute 
...
Vector-Vector Addition W/O RVV:
Vector-Vector Addition took 0.001456 seconds to execute
```

## Notes

- Debug output during `free_tensor_safe()` shows pointer addresses and nullification
