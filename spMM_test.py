import scipy.sparse as sp
import numpy as np
from spMM_CSR import spmm_gustavson_csr

#import triton kernel from file called spMM_CSR.py
from spMM_CSR import spmm_csr_naive_kernel

def main():
    # Create a sparse matrix in CSR format
    A = sp.random(4, 4, density=0.5, format='csr')
    B = np.random.rand(4, 4).astype(np.float32)
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)

    # Call the Triton-based spMM function using Gustavson's algorithm
    spmm_gustavson_csr(A, B, C)

    print("Result from Triton spMM with Gustavson's Algorithm:")
    print(C)

    # Verification using SciPy's built-in sparse matrix multiplication
    C_scipy = A.dot(B)  # SciPy's built-in CSR * dense multiplication

    print("Expected result from SciPy's CSR matrix multiplication:")
    print(C_scipy)

    # Compare the two results
    if np.allclose(C, C_scipy, atol=1e-6):
        print("Triton kernel result matches SciPy's result!")
    else:
        print("Results do not match. Triton kernel may have an issue.")

if __name__ == "__main__":
    main()