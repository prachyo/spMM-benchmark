import cupy as cp
import numpy as np
import scipy.sparse as sp
import ctypes
from numba import cuda

# Load cuSPARSE library
cusparse = ctypes.cdll.LoadLibrary('libcusparse.so')

# Constants for cuSPARSE operations
CUBLAS_OP_N = 0  # Non-transpose operation
CUSPARSE_INDEX_32I = 1
CUSPARSE_INDEX_BASE_ZERO = 0
CUDA_R_32F = 0
CUSPARSE_ALG_DEFAULT = 0  # Default algorithm for CSR

# Create cuSPARSE handle
cusparse_handle = ctypes.c_void_p()
cusparse.cusparseCreate(ctypes.byref(cusparse_handle))

# Convert a SciPy CSR matrix to BSR format
def csr_to_bsr(A_csr, block_size):
    A_csr = A_csr.tocsr()  # Ensure it's in CSR format
    M, N = A_csr.shape

    cusparse = ctypes.cdll.LoadLibrary('libcusparse.so')
    
    # Constants for cuSPARSE operations
    CUSPARSE_DIRECTION_ROW = 0
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0

    # Create cuSPARSE handle
    cusparse_handle = ctypes.c_void_p()
    cusparse.cusparseCreate(ctypes.byref(cusparse_handle))

    # Create matrix descriptor for cuSPARSE
    descr = ctypes.c_void_p()
    cusparse.cusparseCreateMatDescr(ctypes.byref(descr))

    nnz = A_csr.nnz

    # Extract CSR components
    csr_values = A_csr.data.astype(np.float32)
    csr_row_ptr = A_csr.indptr.astype(np.int32)
    csr_col_ind = A_csr.indices.astype(np.int32)

    # Allocate arrays for BSR matrix
    mb = (M + block_size - 1) // block_size
    nb = (N + block_size - 1) // block_size
    bsr_values = np.zeros_like(csr_values)  # Same nnz as CSR
    bsr_row_ptr = np.zeros(mb + 1, dtype=np.int32)
    bsr_col_ind = np.zeros_like(csr_col_ind)

    # Call cuSPARSE's csr2bsr function
    cusparse.cusparseScsr2bsr(
        cusparse_handle,
        CUSPARSE_DIRECTION_ROW,     # Row-major BSR format
        M, N,                       # Matrix dimensions
        descr,                      # Matrix descriptor
        csr_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),    # CSR values
        csr_row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),   # CSR row pointer
        csr_col_ind.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),   # CSR column indices
        block_size,                 # Block size
        descr,                      # Matrix descriptor for BSR
        bsr_values.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),    # BSR values (output)
        bsr_row_ptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),   # BSR row pointer (output)
        bsr_col_ind.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))    # BSR column indices (output)
    )

    # Return the output of conversion from csr to bsr
    return bsr_values, bsr_col_ind, bsr_row_ptr, mb, nb

def cusparse_csrmm(A_csr, B, C):

    # Extract CSR matrix components
    A_data = A_csr.data
    A_indices = A_csr.indices
    A_indptr = A_csr.indptr
    M, K = A_csr.shape  # A is MxK
    _, N = B.shape      # B is KxN

    # Number of non-zero elements in A
    nnz = A_csr.nnz

    # Allocate memory on the GPU using CuPy
    d_A_values = cp.asarray(A_data, dtype=cp.float32)
    d_A_indices = cp.asarray(A_indices, dtype=cp.int32)
    d_A_indptr = cp.asarray(A_indptr, dtype=cp.int32)
    d_B = cp.asarray(B, dtype=cp.float32)
    d_C = cp.zeros((M, N), dtype=cp.float32)  # Allocate space for output matrix C

    # Create matrix descriptor
    descr = ctypes.c_void_p()
    cusparse.cusparseCreateMatDescr(ctypes.byref(descr))

    # Scalars
    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)

    # Leading dimensions
    ldb = B.shape[1]  # Number of columns in B
    ldc = C.shape[1]  # Number of columns in C

    # Create cusparseSpMatDescr (for CSR matrix)
    A_csr_desc = ctypes.c_void_p()
    cusparse.cusparseCreateCsr(
        ctypes.byref(A_csr_desc),
        ctypes.c_int(M),  # rows
        ctypes.c_int(K),  # cols
        ctypes.c_int(nnz),  # nnz
        d_A_indptr.data.ptr,  # rowOffsets
        d_A_indices.data.ptr,  # colInd
        d_A_values.data.ptr,  # values
        ctypes.c_int(CUSPARSE_INDEX_32I),  # rowOffsetsType
        ctypes.c_int(CUSPARSE_INDEX_32I),  # colIndType
        ctypes.c_int(CUSPARSE_INDEX_BASE_ZERO),  # idxBase
        ctypes.c_int(CUDA_R_32F))  # valueType

    # Create dense matrix descriptors for B and C
    B_desc = ctypes.c_void_p()
    cusparse.cusparseCreateDnMat(
        ctypes.byref(B_desc),
        ctypes.c_int(K),  # rows
        ctypes.c_int(N),  # cols
        ctypes.c_int(ldb),  # Leading dimension
        d_B.data.ptr,
        ctypes.c_int(CUDA_R_32F),
        ctypes.c_int(CUSPARSE_INDEX_32I))

    C_desc = ctypes.c_void_p()
    cusparse.cusparseCreateDnMat(
        ctypes.byref(C_desc),
        ctypes.c_int(M),  # rows
        ctypes.c_int(N),  # cols
        ctypes.c_int(ldc),  # Leading dimension
        d_C.data.ptr,
        ctypes.c_int(CUDA_R_32F),
        ctypes.c_int(CUSPARSE_INDEX_32I))

    # Buffer size and buffer allocation
    buffer_size = ctypes.c_size_t()
    cusparse.cusparseSpMM_bufferSize(
        cusparse_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        ctypes.byref(alpha),
        A_csr_desc,
        B_desc,
        ctypes.byref(beta),
        C_desc,
        ctypes.c_int(CUDA_R_32F),
        ctypes.c_int(CUSPARSE_ALG_DEFAULT),  # Correct algorithm for CSR
        ctypes.byref(buffer_size))

    d_buffer = cp.cuda.memory.alloc(buffer_size.value)

    # Perform SpMM (Sparse Matrix x Dense Matrix)
    cusparse.cusparseSpMM(
        cusparse_handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        ctypes.byref(alpha),
        A_csr_desc,
        B_desc,
        ctypes.byref(beta),
        C_desc,
        ctypes.c_int(CUDA_R_32F),
        ctypes.c_int(CUSPARSE_ALG_DEFAULT),  # Correct algorithm for CSR
        d_buffer.ptr)

    # Synchronize the GPU
    #cp.cuda.Device().synchronize()

    # Copy the result back to the host
    # convert C_desc to numpy
    matC_data_ptr = ctypes.c_void_p()
    cusparse.cusparseDnMatGetValues(C_desc, ctypes.byref(matC_data_ptr))
    memptr = cp.cuda.memory.MemoryPointer(cp.cuda.BaseMemory(), matC_data_ptr.value)
    C_gpu = cp.ndarray((M, N), dtype=cp.float32, memptr=memptr)
    C_cpu = cp.asnumpy(C_gpu)

    return C_cpu


def main():
    # Create a sparse matrix in CSR format
    A = sp.random(4, 4, density=0.4, format='csr')
    # Create dense matrix in NumPy called B
    B = np.random.rand(4, 4).astype(np.float32)
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)

    C_actual = A.dot(B)
    print(C_actual)

    # Run the test
    result = cusparse_csrmm(A, B, C)
    print(result)

    print(np.allclose(result, C_actual, atol=1e-6))

if __name__ == '__main__':
    main()

