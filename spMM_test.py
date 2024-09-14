import scipy.sparse as sp
import numpy as np
from spMM_CSR import spmm_gustavson_csr, spmm_gustavson_csr_block, benchmark_spmm, profile_spmm
from spMM_CSR_blocked import benchmark_spmm_block
import time
import cupy as cp

def main():
    # Create a sparse matrix in CSR format
    A = sp.random(1024, 1024, density=0.2, format='csr')
    B = np.random.rand(1024, 1024).astype(np.float32)
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    C_block = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32) # Output matrix for blocked SpMM

    """
    # Call the Triton-based spMM function using Gustavson's algorithm
    gustavson_kernel, C_naive = spmm_gustavson_csr(A, B, C)
    gustavson_block_kernel, C_block = spmm_gustavson_csr_block(A, B, C_block, block_size_rows=2)
    

    print("Result from naive Triton spMM with Gustavson's Algorithm:")
    print(C)

    print("Result from Triton spMM with Gustavson's Algorithm and block processing:")
    print(C_block)

    # Verification using SciPy's built-in sparse matrix multiplication
    C_scipy = A.dot(B)  # SciPy's built-in CSR * dense multiplication

    print("Expected result from SciPy's CSR matrix multiplication:")
    print(C_scipy)

    # Compare the two results
    if np.allclose(C, C_scipy, atol=1e-6):
        print("Triton naive kernel result matches SciPy's result!")
    else:
        print("Results do not match. Triton kernel may have an issue.")

    if np.allclose(C_block, C_scipy, atol=1e-6):
        print("Triton blocked naive kernel result matches SciPy's result!")
    else:
        print("Results do not match. Triton kernel may have an issue.")
    """
    #Run the benchmark
    print("Running benchmark...")
    benchmark_spmm(A, B, C, block_size=128)
    benchmark_spmm_block(A, B, C_block, block_size=64)

    # Perform SpMM: A_csr * B using cuSPARSE via cuPy
    A_csr = cp.sparse.csr_matrix((cp.array(A.data),
                              cp.array(A.indices),
                              cp.array(A.indptr)),
                             shape=A.shape)
    start = time.time()
    B_cp = cp.array(B)
    C_cp = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    C_cp = A_csr.dot(B_cp)  # This internally uses cuSPARSE for sparse-dense matrix multiplication
    cp.cuda.Device().synchronize()  # Ensure all operations are done
    end = time.time()

    # Run the PyTorch Profiler
    #print("\nRunning profiler...")
    #profile_spmm(A, B, C, block_size=128)

    execution_time_s = end - start

    # Calculate FLOPs (only for non-zero elements in A)
    nnz_A = len(A.data)  # Number of non-zero elements in sparse matrix A
    FLOP_count = 2 * nnz_A * B.shape[1]  # Each non-zero in A performs 2*N operations (1 mul, 1 add)

    # Calculate FLOP/s
    FLOP_s = FLOP_count / execution_time_s  # FLOP/s

    # Calculate memory bandwidth
    bytes_transferred = (A_csr.data.nbytes + A.indices.nbytes + A.indptr.nbytes +
                         B.nbytes + C.nbytes)
    memory_bandwidth = bytes_transferred / execution_time_s  # B/s

    print(f"\ncuSPARSE SpMM Metrics:")
    print(f"Execution Time: {execution_time_s:.6f} seconds")
    print(f"FLOP/s: {FLOP_s:.2f} FLOP/s")
    print(f"Memory Bandwidth: {memory_bandwidth:.2f} B/s")

if __name__ == "__main__":
    main()