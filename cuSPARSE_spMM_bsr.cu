#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::cerr << "CUDA API failed at line " << __LINE__ << " with error: " \
                  << cudaGetErrorString(status) << " (" << status << ")\n";    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::cerr << "CUSPARSE API failed at line " << __LINE__ << " with error: " \
                  << cusparseGetErrorString(status) << " (" << status << ")\n";\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

#define CHECK_CUBLAS(func)                                                     \
{                                                                              \
    cublasStatus_t status = (func);                                            \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
        std::cerr << "cuBLAS API failed at line " << __LINE__ << " with error: " \
                  << status << "\n";                                           \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

extern "C" void benchmark_cusparseSpMMBSR(int A_num_rows, int A_num_cols, int A_nnz, int B_num_cols, int block_size,
                               const int* hA_csrOffsets, const int* hA_columns,
                               const float* hA_values, const float* hB,
                               float* hC) {
    // Convert arrays to vectors
    std::vector<int> hA_csrOffsets_vec(hA_csrOffsets, hA_csrOffsets + A_num_rows + 1);
    std::vector<int> hA_columns_vec(hA_columns, hA_columns + A_nnz);
    std::vector<float> hA_values_vec(hA_values, hA_values + A_nnz);
    std::vector<float> hB_vec(hB, hB + A_num_cols * B_num_cols);

    // Device memory allocation
    int *dA_csrOffsets, *dA_columns, *dA_bsrOffsets = nullptr, *dA_bsrColumns = nullptr;
    float *dA_values, *dA_bsrValues = nullptr, *dB, *dC;
    CHECK_CUDA(cudaMalloc((void**)&dA_csrOffsets, (A_num_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_columns, A_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_values, A_nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dB, A_num_cols * B_num_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dC, A_num_rows * B_num_cols * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(dA_csrOffsets, hA_csrOffsets_vec.data(), (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns_vec.data(), A_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values_vec.data(), A_nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB_vec.data(), A_num_cols * B_num_cols * sizeof(float), cudaMemcpyHostToDevice));

    // cuSPARSE handle and matrix descriptor initialization
    cusparseHandle_t handle;
    cusparseMatDescr_t descr;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));

    // Convert CSR to BSR
    int mb = (A_num_rows + block_size - 1) / block_size;
    int nb = (A_num_cols + block_size - 1) / block_size;
    int nnzb;
    CHECK_CUDA(cudaMalloc((void**)&dA_bsrOffsets, (mb + 1) * sizeof(int)));
    CHECK_CUSPARSE(cusparseXcsr2bsrNnz(handle, CUSPARSE_DIRECTION_ROW, A_num_rows, A_num_cols, descr,
                                       dA_csrOffsets, dA_columns, block_size, descr, dA_bsrOffsets, &nnzb));

    CHECK_CUDA(cudaMalloc((void**)&dA_bsrColumns, nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dA_bsrValues, nnzb * block_size * block_size * sizeof(float)));

    CHECK_CUSPARSE(cusparseScsr2bsr(handle, CUSPARSE_DIRECTION_ROW, A_num_rows, A_num_cols, descr,
                                    dA_values, dA_csrOffsets, dA_columns, block_size, descr,
                                    dA_bsrValues, dA_bsrOffsets, dA_bsrColumns));

    // Perform BSRMM
    float alpha = 1.0f;
    float beta = 0.0f;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    CHECK_CUSPARSE(cusparseSbsrmm(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE, mb, B_num_cols, nb, nnzb, &alpha,
                                  descr, dA_bsrValues, dA_bsrOffsets, dA_bsrColumns, block_size,
                                  dB, A_num_cols, &beta, dC, A_num_rows));

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result from device to host
    CHECK_CUDA(cudaMemcpy(hC, dC, A_num_rows * B_num_cols * sizeof(float), cudaMemcpyDeviceToHost));

    // Clean up and free device memory
    CHECK_CUDA(cudaFree(dA_csrOffsets));
    CHECK_CUDA(cudaFree(dA_columns));
    CHECK_CUDA(cudaFree(dA_values));
    CHECK_CUDA(cudaFree(dA_bsrOffsets));
    CHECK_CUDA(cudaFree(dA_bsrColumns));
    CHECK_CUDA(cudaFree(dA_bsrValues));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));

    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    float execution_time_s = milliseconds / 1000.0f;

    // Calculate GFLOPs and memory bandwidth
    float gflops = (2.0f * A_nnz * B_num_cols) / execution_time_s / 1e9f;
    float bytes_transferred = (A_nnz * sizeof(float) + A_nnz * sizeof(int) + A_num_rows * sizeof(int) + A_num_cols * B_num_cols * sizeof(float) + A_num_rows * B_num_cols * sizeof(float));
    float memory_bandwidth = bytes_transferred / execution_time_s / 1e9f;

    printf("\n-----cuSPARSE BSR SpMM Metrics:\n");
    printf("Execution time: %f seconds\n", execution_time_s);
    printf("GFLOP/s: %f\n", gflops);
    printf("Memory Bandwidth: %.2f GB/s\n", memory_bandwidth);
}

void matrixMultiplyCUBLAS(int A_num_rows, int A_num_cols, int B_num_cols, const float* hA, const float* hB, float* hC) {
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc((void**)&dA, A_num_rows * A_num_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dB, A_num_cols * B_num_cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dC, A_num_rows * B_num_cols * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA, hA, A_num_rows * A_num_cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, A_num_cols * B_num_cols * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, B_num_cols, A_num_rows, A_num_cols, &alpha, dB, B_num_cols, dA, A_num_cols, &beta, dC, B_num_cols));

    CHECK_CUDA(cudaMemcpy(hC, dC, A_num_rows * B_num_cols * sizeof(float), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUBLAS(cublasDestroy(handle));
}

int main() {
    // Host problem definition
    int A_num_rows = 4;
    int A_num_cols = 4;
    int A_nnz = 9;
    int B_num_cols = 3;
    int block_size = 2;

    int hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    int hA_columns[] = { 0, 1, 2, 2, 0, 1, 3, 2, 3 };
    float hA_values[] = { 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0 };
    float hB[] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };
    float hC[12] = { 0 };
    float hC_cublas[12] = { 0 };

    // Call the benchmark function
    benchmark_cusparseSpMMBSR(A_num_rows, A_num_cols, A_nnz, B_num_cols, block_size,
                              hA_csrOffsets, hA_columns, hA_values, hB, hC);

    // Perform matrix multiplication using cuBLAS
    std::vector<float> hA_dense(A_num_rows * A_num_cols, 0.0f);
    for (int i = 0; i < A_num_rows; ++i) {
        for (int j = hA_csrOffsets[i]; j < hA_csrOffsets[i + 1]; ++j) {
            hA_dense[i * A_num_cols + hA_columns[j]] = hA_values[j];
        }
    }
    matrixMultiplyCUBLAS(A_num_rows, A_num_cols, B_num_cols, hA_dense.data(), hB, hC_cublas);

    // Print result from benchmark function
    std::cout << "Result matrix C from benchmark_cusparseSpMMBSR:" << std::endl;
    for (int i = 0; i < A_num_rows; ++i) {
        for (int j = 0; j < B_num_cols; ++j) {
            std::cout << hC[i + j * A_num_rows] << " ";
        }
        std::cout << std::endl;
    }

    // Print result from cuBLAS
    std::cout << "Result matrix C from cuBLAS:" << std::endl;
    for (int i = 0; i < A_num_rows; ++i) {
        for (int j = 0; j < B_num_cols; ++j) {
            std::cout << hC_cublas[i + j * A_num_rows] << " ";
        }
        std::cout << std::endl;
    }

    // Compare results
    bool match = true;
    for (int i = 0; i < A_num_rows * B_num_cols; ++i) {
        if (std::fabs(hC[i] - hC_cublas[i]) > 1e-5) {
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
    }

    return EXIT_SUCCESS;
}