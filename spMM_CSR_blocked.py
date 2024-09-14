import numpy as np
import torch
import triton
import triton.language as tl
import time

@triton.jit
def spmm_gustavson_csr_block_kernel(
    data_ptr,          # Pointer to non-zero values of sparse matrix A
    indices_ptr,       # Pointer to column indices for non-zero values of sparse matrix A
    indptr_ptr,        # Pointer to row pointers of sparse matrix A
    B_ptr,             # Pointer to dense matrix B
    C_ptr,             # Pointer to result matrix C
    N: tl.constexpr,   # Feature size for dense matrix, should be a compile-time constant
    stride_B,          # Stride for dense matrix B (row-major)
    stride_C,          # Stride for dense matrix C (row-major)
    BLOCK_SIZE_ROWS: tl.constexpr  # Number of rows in each block
):
    # Get the block index for rows
    block_row_idx = tl.program_id(0) * BLOCK_SIZE_ROWS

    # Loop over the rows in this block
    for row_offset in range(BLOCK_SIZE_ROWS):
        row_idx = block_row_idx + row_offset

        # Fetch the row start and end from the CSR indptr
        row_start = tl.load(indptr_ptr + row_idx)
        row_end = tl.load(indptr_ptr + row_idx + 1)

        # Initialize the result row for C (dense row)
        result = tl.zeros((N,), dtype=tl.float32)

        # Loop over non-zero elements in row
        for i in range(row_start, row_end):
            val_A = tl.load(data_ptr + i)
            col_A = tl.load(indices_ptr + i)

            # Perform matrix multiplication over the columns in B
            j = tl.arange(0, N)
            val_B = tl.load(B_ptr + col_A * stride_B + j)
            result += val_A * val_B

        # Store the result row back into the dense matrix C
        tl.store(C_ptr + row_idx * stride_C + tl.arange(0, N), result)

def spmm_gustavson_csr_block(A_csr, B, C, block_size_rows=8):
    # A_csr: sparse matrix in CSR format
    # B: dense matrix (NumPy array)
    # C: output matrix (NumPy array)

    # Extract components of the CSR matrix A
    data = A_csr.data  # Non-zero values
    indices = A_csr.indices  # Column indices of non-zero values
    indptr = A_csr.indptr  # Row pointers

    M, K = A_csr.shape  # A is MxK (M rows, K columns)
    _, N = B.shape  # B is KxN (K rows, N columns)

    # Allocate GPU memory using PyTorch and copy data
    device = torch.device('cuda')

    # Convert NumPy arrays to PyTorch tensors and move them to GPU
    data_ptr = torch.tensor(data, dtype=torch.float32, device=device)
    indices_ptr = torch.tensor(indices, dtype=torch.int32, device=device)
    indptr_ptr = torch.tensor(indptr, dtype=torch.int32, device=device)
    B_ptr = torch.tensor(B, dtype=torch.float32, device=device)
    C_ptr = torch.zeros_like(torch.tensor(C, dtype=torch.float32, device=device))  # Create a zero-initialized tensor for C

    # Define the grid size (blocks of rows)
    grid = ((M + block_size_rows - 1) // block_size_rows,)   # This ensures each block of rows has one grid block

    def kernel_launch():
        # Launch the Triton kernel
        spmm_gustavson_csr_block_kernel[grid](
            data_ptr, indices_ptr, indptr_ptr,
            B_ptr, C_ptr,
            N,
            BLOCK_SIZE_ROWS=block_size_rows,
            stride_B=B.shape[1],
            stride_C=C.shape[1]
        )
        torch.cuda.synchronize()
    return kernel_launch, C_ptr

def benchmark_spmm_block(A_csr, B, C, block_size=128):
    kernel_launch, C_ptr = spmm_gustavson_csr_block(A_csr, B, C, block_size)

    # Perform SpMM: A_csr * B using cuSPARSE via cuPy
    C_actual = A_csr.dot(B)  # Expected result from SciPy's CSR * dense multiplication

    # Calculate FLOPs (only for non-zero elements in A)
    nnz_A = len(A_csr.data)  # Number of non-zero elements in sparse matrix A
    FLOP_count = 2 * nnz_A * B.shape[1]  # Each non-zero in A performs 2*N operations (1 mul, 1 add)

    # Launch Triton kernel (or any CUDA operation)
    start = time.time()
    kernel_launch()

    # Synchronize and end event
    torch.cuda.synchronize()
    end = time.time()

    # Calculate execution time
    execution_time_s = end - start

    # Calculate FLOP/s
    FLOP_s = FLOP_count / execution_time_s  # FLOP/s
    GFLOP_s = FLOP_s / 1e9 # GFLOP/s

    # Calculate memory bandwidth
    bytes_transferred = (A_csr.data.nbytes + A_csr.indices.nbytes + A_csr.indptr.nbytes +
                         B.nbytes + C.nbytes)
    memory_bandwidth = (bytes_transferred / execution_time_s) / 1e9 # GB/s

    # Copy result back to CPU
    C[:] = C_ptr.cpu().numpy()

    # Print the results
    print(f"\nTriton Blocked SpMM Metrics:")
    print("Is soln correct? ", np.allclose(C, C_actual, atol=1e-6))
    print(f"Execution Time: {execution_time_s:.6f} seconds")
    print(f"GFLOP/s: {GFLOP_s:.2f} GFLOP/s")
    print(f"Memory Bandwidth: {memory_bandwidth:.2f} GB/s")

    