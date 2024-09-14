import scipy.sparse as sp
import numpy as np
from spMM_CSR import spmm_gustavson_csr, spmm_gustavson_csr_block

def main():
    # Create a sparse matrix in CSR format
    A = sp.random(4, 4, density=0.5, format='csr')
    B = np.random.rand(4, 4).astype(np.float32)
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    C_block = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)


    # Call the Triton-based spMM function using Gustavson's algorithm
    spmm_gustavson_csr(A, B, C)
    spmm_gustavson_csr_block(A, B, C_block, block_size_rows=2)

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

if __name__ == "__main__":
    main()