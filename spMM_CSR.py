import torch
import triton
import triton.language as tl
import numpy as np
from scipy.sparse import random, csr_matrix

@triton.jit
def spmm_csr_naive_kernel(values, col_indices, row_ptr, B, C, feature_size: tl.constexpr):
    row_id = tl.program_id(0)  # Get the row index for this block

    row_start = tl.load(row_ptr + row_id)  # Start index of the row in values and col_indices
    row_end = tl.load(row_ptr + row_id + 1)  # End index of the row in values and col_indices

    # Load col_indices and values for the current row
    col_indices_slice = tl.load(col_indices + row_start + tl.arange(0, row_end - row_start))
    values_slice = tl.load(values + row_start + tl.arange(0, row_end - row_start))

    for i in range(row_end - row_start):
        for j in range(feature_size):
            a_val = values_slice[i]
            b_val = tl.load(B + col_indices_slice[i] * feature_size + j) # Value from dense matrix B
            c_val = tl.load(C + row_id * feature_size + j)  # Current value in output matrix C
            c_val += a_val * b_val  # Multiply and accumulate
            tl.store(C + row_id * feature_size + j, c_val)  # Store the result back to C

@triton.jit
def spmm_csr_blocked_kernel(
    values,
    col_indices,
    row_ptr,
    B,
    C,
    # The block size
    BLOCK_SIZE_M: tl.constexpr,
    feature_size: tl.constexpr
):
    
    pid_m = tl.program_id(0)
    offset = pid_m * BLOCK_SIZE_M
    
    # Create an array to store row pointers for the block
    row_ptrs = tl.zeros([BLOCK_SIZE_M * 2], dtype=tl.int32)
    
    for i in range(BLOCK_SIZE_M):
        row_ptrs[i][0] = tl.load(row_ptr + offset + i)
        row_ptrs[i][1] = tl.load(row_ptr + offset + i + 1)

    # Loop over every row to apply Gustavson's method
    for i in range(BLOCK_SIZE_M):
        row_start = row_ptrs[i][0]
        row_end = row_ptrs[i][1]

        # Load col_indices and values for the current row
        col_indices_slice = tl.load(col_indices + row_start + tl.arange(0, row_end - row_start))
        values_slice = tl.load(values + row_start + tl.arange(0, row_end - row_start))

        for j in range(row_end - row_start):
            for k in range(feature_size):
                a_val = values_slice[j]
                b_val = tl.load(B + feature_size * col_indices_slice[j] + k)
                c_val = tl.load(C + feature_size * (offset + i) + k)
                c_val += a_val * b_val  # Multiply and accumulate
                tl.store(C + feature_size * (offset + i) + k, c_val) # Store back the result


    

# Driver code for testing naive spmm kernel
n_rows = 100  # Number of rows
n_cols = 100 # Number of columns
density = 0.2  # Fraction of non-zero elements (20% non-zero)

# generate random sparse matrix
random_sparse = random(n_rows, n_cols, density=density, format='csr', dtype=np.float32)

# Convert the sparse matrix to CSR format
csr_matrix_rep = csr_matrix(random_sparse)

# Generate random dense matrix
B = torch.rand((n_cols, n_rows), dtype=torch.float32, device='cuda')

# Prepare the output matrix
C = torch.zeros((n_rows, n_cols), dtype=torch.float32, device='cuda')

# Launch the kernel
grid = (n_rows,)
spmm_csr_naive_kernel[grid](csr_matrix_rep.data, csr_matrix_rep.indices, csr_matrix_rep.indptr, B, C, n_cols)

# Retrieve and print the result
print(C)
