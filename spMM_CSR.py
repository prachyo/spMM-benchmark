import torch
import triton
import triton.language as tl
import numpy as np
from scipy.sparse import random, csr_matrix
from torch.profiler import profile, record_function, ProfilerActivity
import time

# Triton kernel for SpMM using Gustavson's Algorithm (CSR format)
@triton.jit
def spmm_gustavson_csr_kernel(
    data_ptr,          # Pointer to non-zero values of sparse matrix A
    indices_ptr,       # Pointer to column indices for non-zero values of sparse matrix A
    indptr_ptr,        # Pointer to row pointers of sparse matrix A
    B_ptr,             # Pointer to dense matrix B
    C_ptr,             # Pointer to result matrix C
    M, N: tl.constexpr, K,           # Ensure N is a compile-time constant
    stride_B,          # Stride for dense matrix B (row-major)
    stride_C,          # Stride for dense matrix C (row-major)
    BLOCK_SIZE: tl.constexpr
):
    # Each Triton program computes one row of C
    row_idx = tl.program_id(0)  # Get the row this program is responsible for

    # Fetch the row start and end from the CSR indptr
    row_start = tl.load(indptr_ptr + row_idx)
    row_end = tl.load(indptr_ptr + row_idx + 1)

    # Initialize the result row for C (dense row)
    result = tl.zeros((N,), dtype=tl.float32)  # Ensure N is compile-time constant

    # Loop over non-zero elements in row
    for i in range(row_start, row_end):
        val_A = tl.load(data_ptr + i)
        col_A = tl.load(indices_ptr + i)

        # Parallelize over columns of matrix B corresponding to col_A
        j = tl.arange(0, N)  # Create a parallelized index array for columns
        val_B = tl.load(B_ptr + col_A * stride_B + j)
        result += val_A * val_B

    # Store the result row back into the dense matrix C
    tl.store(C_ptr + row_idx * stride_C + tl.arange(0, N), result)

@triton.jit
def spmm_gustavson_csr_block_kernel(
    data_ptr,          # Pointer to non-zero values of sparse matrix A
    indices_ptr,       # Pointer to column indices for non-zero values of sparse matrix A
    indptr_ptr,        # Pointer to row pointers of sparse matrix A
    B_ptr,             # Pointer to dense matrix B
    C_ptr,             # Pointer to result matrix C
    M, N: tl.constexpr, K,           # N should be a compile-time constant
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

def spmm_gustavson_csr(A_csr, B, C, block_size=128):
    # A_csr: sparse matrix in CSR format
    # B: dense matrix
    # C: output matrix

    # Extract components of the CSR matrix A
    data = A_csr.data  # Non-zero values
    indices = A_csr.indices  # Column indices of non-zero values
    indptr = A_csr.indptr  # Row pointers

    M, K = A_csr.shape  # A is MxK
    _, N = B.shape  # B is KxN

    # Allocate GPU memory using PyTorch and copy data
    device = torch.device('cuda')

    # Convert NumPy arrays to PyTorch tensors and move them to GPU
    data_ptr = torch.tensor(data, dtype=torch.float32, device=device)
    indices_ptr = torch.tensor(indices, dtype=torch.int32, device=device)
    indptr_ptr = torch.tensor(indptr, dtype=torch.int32, device=device)
    B_ptr = torch.tensor(B, dtype=torch.float32, device=device)
    C_ptr = torch.tensor(C, dtype=torch.float32, device=device)

    # Call Triton kernel using Gustavson's algorithm
    grid = lambda meta: (M,)

    def kernel_launch():
        spmm_gustavson_csr_kernel[grid](
            data_ptr, indices_ptr, indptr_ptr,
            B_ptr, C_ptr,
            M, N, K,
            BLOCK_SIZE=block_size,
            stride_B=B.shape[1],
            stride_C=C.shape[1]  # Removed stride_A, since it's not used
        )
        torch.cuda.synchronize()  # Ensure GPU operations complete
    return kernel_launch, C_ptr

def benchmark_spmm(A_csr, B, C, block_size=128):
    kernel_launch, C_ptr = spmm_gustavson_csr(A_csr, B, C, block_size)

    """"
    # Use Triton Benchmark to time the kernel
    bench = triton.testing.Benchmark(kernel_launch)
    execution_time = bench.run()  # Measure execution time
    """

    # Calculate FLOPs (only for non-zero elements in A)
    nnz_A = len(A_csr.data)  # Number of non-zero elements in sparse matrix A
    FLOP_count = 2 * nnz_A * B.shape[1]  # Each non-zero in A performs 2*N operations (1 mul, 1 add)

    #start_event = torch.cuda.Event(enable_timing=True)
    #end_event = torch.cuda.Event(enable_timing=True)

    # Record start event
    #start_event.record()

    # Launch Triton kernel (or any CUDA operation)
    start = time.time()
    kernel_launch()

    # Record end event and synchronize
    #end_event.record()
    torch.cuda.synchronize()
    end = time.time()

    # Measure elapsed time in milliseconds
    #execution_time_ms = start_event.elapsed_time(end_event)
    #execution_time_s = execution_time_ms / 1000  # Convert to seconds
    execution_time_s = end - start

    # Calculate FLOP/s
    FLOP_s = FLOP_count / execution_time_s  # FLOP/s
    GFLOP_s = FLOP_s / 1e9 # GFLOP/s

    # Calculate memory bandwidth
    bytes_transferred = (A_csr.data.nbytes + A_csr.indices.nbytes + A_csr.indptr.nbytes +
                         B.nbytes + C.nbytes)
    memory_bandwidth = bytes_transferred / execution_time_s  # B/s

    # Print the results
    print(f"\nTriton SpMM Metrics:")
    print(f"Execution Time: {execution_time_s:.6f} seconds")
    print(f"FLOP/s: {FLOP_s:.2f} FLOP/s")
    print(f"Memory Bandwidth: {memory_bandwidth:.2f} B/s")

    # Copy result back to CPU
    C[:] = C_ptr.cpu().numpy()

# Use PyTorch Profiler to capture detailed metrics (like memory usage and GPU utilization)
def profile_spmm(A_csr, B, C, block_size=128):
    kernel_launch, C_ptr = spmm_gustavson_csr(A_csr, B, C, block_size)

    # Use PyTorch Profiler to profile Triton kernel
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        with record_function("Triton SpMM Kernel"):
            kernel_launch()

    # Display profiling results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # Manually compute additional metrics
    total_cuda_time = prof.key_averages().total_average().cuda_time_total / 1e6  # in seconds

    # Calculate FLOPs (only for non-zero elements in A using Gustavson's algorithm)
    nnz_A = len(A_csr.data)  # Number of non-zero elements in sparse matrix A
    FLOP_count = 2 * nnz_A * B.shape[1]  # Each non-zero in A performs 2*N operations (1 mul, 1 add)

    # Calculate FLOP/s
    GFLOP_s = (FLOP_count / total_cuda_time) / 1e9

    # Calculate memory bandwidth
    bytes_transferred = (A_csr.data.nbytes + A_csr.indices.nbytes + A_csr.indptr.nbytes +
                         B.nbytes + C.nbytes)
    memory_bandwidth = bytes_transferred / total_cuda_time / 1e9  # GB/s

    # Print the additional metrics
    print(f"\nTriton SpMM Metrics:")
    print(f"Execution Time: {total_cuda_time:.6f} seconds")
    print(f"GFLOP/s: {GFLOP_s:.2f} GFLOP/s")
    print(f"Memory Bandwidth: {memory_bandwidth:.2f} GB/s")

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

    # Launch the Triton kernel
    spmm_gustavson_csr_block_kernel[grid](
        data_ptr, indices_ptr, indptr_ptr,
        B_ptr, C_ptr,
        M, N, K,
        BLOCK_SIZE_ROWS=block_size_rows,
        stride_B=B.shape[1],
        stride_C=C.shape[1]
    )

    # Copy the result back to CPU and overwrite the NumPy array C
    C[:] = C_ptr.cpu().numpy()