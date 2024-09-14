import ctypes
import time
import numba
import numpy as np
from scipy import sparse as sp
from numba import cuda
from numba.cuda.cudadrv.libs import open_cudalib
from pycuda.autoinit import context  # to initialize CUDA context

# Load cuSPARSE library
cusparse = open_cudalib('cusparse')

# cuSPARSE helper functions and constants
CUBLAS_OP_N = 0

def check_cusparse_status(status):
    if status != 0:
        raise RuntimeError(f"cuSPARSE error: {status}")

def blocked_spmm_cusparse(A_csr, B, C, block_size):
    """
    Perform Blocked Sparse Matrix (CSR) x Dense Matrix Multiplication using cuSPARSE.

    Parameters:
    - A_csr: SciPy CSR sparse matrix (sparse matrix A)
    - B: NumPy dense matrix (matrix B)
    - C: NumPy dense matrix (result matrix C, initialized with zeros)
    - block_size: Size of the block for blocking (BSR format)
    """

    # Extract CSR matrix components (from SciPy CSR matrix)
    A_data = A_csr.data
    A_indices = A_csr.indices
    A_indptr = A_csr.indptr
    M, K = A_csr.shape  # A is MxK
    _, N = B.shape  # B is KxN
    nnz = A_csr.nnz  # Number of non-zero elements in A

    # Allocate memory on GPU using Numba
    d_A_values = cuda.to_device(A_data)
    d_A_indices = cuda.to_device(A_indices)
    d_A_indptr = cuda.to_device(A_indptr)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)

    # Create cuSPARSE handle
    handle = ctypes.c_void_p()
    cusparse.cusparseCreate(ctypes.byref(handle))

    # Convert CSR to BSR (blocked sparse row format)
    # Create matrix descriptor for CSR matrix
    descr = ctypes.c_void_p()
    cusparse.cusparseCreateMatDescr(ctypes.byref(descr))

    # Calculate BSR dimensions
    mb = (M + block_size - 1) // block_size  # Number of row blocks
    nb = (K + block_size - 1) // block_size  # Number of column blocks
    nnzb = ctypes.c_int()  # Number of non-zero blocks in BSR

    # Allocate device memory for BSR matrix (nnzb * block_size * block_size)
    d_A_bsr_values = cuda.device_array(nnzb.value * block_size * block_size, dtype=np.float32)
    d_A_bsr_indices = cuda.device_array_like(d_A_indices)
    d_A_bsr_indptr = cuda.device_array(mb + 1, dtype=np.int32)

    # Convert CSR to BSR
    status = cusparse.cusparseScsr2bsr(handle,
                              CUBLAS_OP_N,
                              M, K, descr,
                              d_A_values.device_ctypes_pointer.value,
                              d_A_indptr.device_ctypes_pointer.value,
                              d_A_indices.device_ctypes_pointer.value,
                              block_size,
                              descr,
                              d_A_bsr_values.device_ctypes_pointer.value,
                              d_A_bsr_indptr.device_ctypes_pointer.value,
                              d_A_bsr_indices.device_ctypes_pointer.value)

    # Check if conversion was successful
    check_cusparse_status(status)

    # Perform Blocked SpMM (BSR x Dense)
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)
    ldb = N
    ldc = N

    # Call the cuSPARSE BSRMM function (float32)
    status = cusparse.cusparseSbsrmm(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            mb, N, nb,
                            nnzb.value, ctypes.byref(alpha), descr,
                            d_A_bsr_values.device_ctypes_pointer.value,
                            d_A_bsr_indptr.device_ctypes_pointer.value,
                            d_A_bsr_indices.device_ctypes_pointer.value,
                            block_size, d_B.device_ctypes_pointer.value,
                            ldb, ctypes.byref(beta),
                            d_C.device_ctypes_pointer.value, ldc)

    # Perform Blocked SpMM (BSR x Dense)
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)
    ldb = N
    ldc = N

    # Call the cuSPARSE BSRMM function (float32)
    cusparse.cusparseSbsrmm(handle,
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            mb, N, nb,
                            nnzb, ctypes.byref(alpha), descr,
                            d_A_bsr_values.device_ctypes_pointer.value,
                            d_A_bsr_indptr.device_ctypes_pointer.value,
                            d_A_bsr_indices.device_ctypes_pointer.value,
                            block_size, d_B.device_ctypes_pointer.value,
                            ldb, ctypes.byref(beta),
                            d_C.device_ctypes_pointer.value, ldc)
    

    if d_C is None:
        raise ValueError("GPU array d_C is not initialized.")
    
    # Synchronize
    cuda.synchronize()

    # Copy result back to host
    d_C.copy_to_host(C)

    # Destroy cuSPARSE handle and descriptors
    cusparse.cusparseDestroy(handle)
    cusparse.cusparseDestroyMatDescr(descr)

    return C

def benchmark_cuSPARSE_blocked(A_csr, B, C, block_size):

    # new numpy array that copies the original C
    C_copy = np.copy(C)

    # Perform SpMM: A_csr * B using cuSPARSE via cuPy
    C_actual = A_csr.dot(B)  # Expected result from SciPy's CSR * dense multiplication

    # Calculate FLOPs (only for non-zero elements in A)
    nnz_A = len(A_csr.data)  # Number of non-zero elements in sparse matrix A
    FLOP_count = 2 * nnz_A * B.shape[1]  # Each non-zero in A performs 2*N operations (1 mul, 1 add)

    # Launch Triton kernel (or any CUDA operation)
    start = time.time()
    C_result = blocked_spmm_cusparse(A_csr, B, C_copy, block_size)

    # Synchronize and end event
    cuda.synchronize()
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

    # Copy contents of C_result to C
    C[:] = C_result

    # Print the results
    print(f"\ncuSPARSE Blocked SpMM Metrics:")
    print("Is soln correct? ", np.allclose(C, C_actual, atol=1e-6))
    print(f"Execution Time: {execution_time_s:.6f} seconds")
    print(f"GFLOP/s: {GFLOP_s:.2f} GFLOP/s")
    print(f"Memory Bandwidth: {memory_bandwidth:.2f} GB/s")


