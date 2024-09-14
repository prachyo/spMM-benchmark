import ctypes
import numba
import scipy.sparse as sp
import numpy as np
from spMM_CSR import benchmark_spmm, profile_spmm
from spMM_CSR_blocked import benchmark_spmm_block
from spMM_cuSPARSE_blocked import benchmark_cuSPARSE_blocked
from spMM_cuSPARSE_csrmm import benchmark_cuSPARSE_csrmm
import time
import cupy as cp

def benchmark_cuSPARSE(A, B, C):
    # A is a sparse matrix in CSR format from SciPy
    # B is a dense matrix from NumPy
    # C is the output matrix from NumPy

    # Perform SpMM: A_csr * B using cuSPARSE via cuPy
    C_actual = A.dot(B)  # Expected result from SciPy's CSR * dense multiplication

    A_csr = cp.sparse.csr_matrix((cp.array(A.data),
                                  cp.array(A.indices),
                                  cp.array(A.indptr)),
                                 shape=A.shape)

    B_cp = cp.array(B)
    C_cp = cp.array(C)

    # Start event for cuSPARSE spMM
    start = time.time()

    # Run event and synchronize
    C_cp = A_csr.dot(B_cp)  # This internally uses cuSPARSE for sparse-dense matrix multiplication
    cp.cuda.Device().synchronize()  # Ensure all operations are done
    end = time.time()

    # Convert C_cp to NumPy array
    C = C_cp.get()

    execution_time_s = end - start

    # Calculate FLOPs (only for non-zero elements in A)
    nnz_A = len(A.data)  # Number of non-zero elements in sparse matrix A
    FLOP_count = 2 * nnz_A * B.shape[1]  # Each non-zero in A performs 2*N operations (1 mul, 1 add)

    # Calculate FLOP/s
    FLOP_s = FLOP_count / execution_time_s  # FLOP/s
    GFLOP_s = FLOP_s / 1e9  # GFLOP/s

    # Calculate memory bandwidth
    bytes_transferred = (A_csr.data.nbytes + A.indices.nbytes + A.indptr.nbytes +
                         B.nbytes + C.nbytes)
    memory_bandwidth = (bytes_transferred / execution_time_s) / 1e9 # GB/s



    print(f"\ncuSPARSE SpMM Metrics:")
    print("Is soln correct? ", np.allclose(C, C_actual, atol=1e-6))
    print(f"Execution Time: {execution_time_s:.6f} seconds")
    print(f"GFLOP/s: {GFLOP_s:.2f} GFLOP/s")
    print(f"Memory Bandwidth: {memory_bandwidth:.2f} GB/s")

def main():

    # Load the shared library
    cusparse_bsr = ctypes.CDLL('./libcuSPARSE_spMM_bsr.so')

    # Create a sparse matrix in CSR format
    A = sp.random(2048, 2048, density=0.4, format='csr')
    B = np.random.rand(2048, 2048).astype(np.float32)
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    C_block = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32) # Output matrix for blocked SpMM

    #Run the benchmark
    print("Running benchmarks...")
    benchmark_spmm(A, B, C)
    benchmark_spmm_block(A, B, C_block, block_size=128)
    benchmark_cuSPARSE(A, B, C)
    try:
        cusparse_bsr.benchmark_cusparseSpMMBSR(
            ctypes.c_int(A.shape[0]),
            ctypes.c_int(A.shape[1]),
            ctypes.c_int(A.nnz),
            ctypes.c_int(B.shape[1]),
            ctypes.c_int(128),  # Block size
            ctypes.c_void_p(A.indptr.ctypes.data),
            ctypes.c_void_p(A.indices.ctypes.data),
            ctypes.c_void_p(A.data.ctypes.data),
            ctypes.c_void_p(B.ctypes.data),
            ctypes.c_void_p(C.ctypes.data),
        )
    except Exception as e:
        print(f"Error calling cuSPARSE_bsr function: {e}")
    #benchmark_cuSPARSE_csrmm(A, B, C)
    #benchmark_cuSPARSE_blocked(A, B, C_block, block_size=128)

    return
    

if __name__ == "__main__":
    main()