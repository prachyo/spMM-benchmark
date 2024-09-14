#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <random>
#include <fstream>
#include <assert.h>
#include <cuda_runtime.h>
#include "cusparse.h"

std::mt19937_64 gen(1234);

template<typename T>
T* vec2ptr(std::vector<T> v) {
    T* ptr = (T*) malloc(v.size() * sizeof(T));
    for (int i = 0; i < v.size(); ++i) {
        ptr[i] = v[i];
    }
    return ptr;
}

int randomBSRMatrix(int mb, int nb, int blockDim, float p, int** hostBsrRowPtr, int** hostBsrColInd, float** hostBsrVal, float minVal=-1, float maxVal=1) {
    std::uniform_real_distribution<float> flip(0, 1), dist(minVal, maxVal);
    int blockNum = blockDim * blockDim;
    *hostBsrRowPtr = (int*) malloc((mb + 1) * sizeof(int));
    int cnt = 0;
    (*hostBsrRowPtr)[0] = cnt;
    std::vector<int> indices;
    std::vector<float> vals;
    for (int i = 1; i <= mb; ++i) {
        for (int j = 0; j < nb; ++j) {
            if (flip(gen) < p) {
                indices.push_back(j);
                for (int k = 0; k < blockNum; ++k) {
                    vals.push_back(dist(gen));
                }
                ++cnt;
            }
        }
        (*hostBsrRowPtr)[i] = cnt;
        // if (i % 1000 == 0) {
        //     printf("i = %d\n", i);
        // }
    }
    *hostBsrColInd = vec2ptr(std::move(indices));
    *hostBsrVal = vec2ptr(std::move(vals));

    // Generating random BSR matrix may be time-consuming, so we record it for next time use.
    // std::string bd = std::to_string(blockDim);
    // std::fstream s1("bsr_" + bd + "_indptr.txt", std::ios::out | std::ios::trunc);
    // std::fstream s2("bsr_" + bd + "_indices.txt", std::ios::out | std::ios::trunc);

    // s1 << mb + 1 << std::endl;
    // for (int i = 0; i <= mb; ++i) {
    //     s1 << (*hostBsrRowPtr)[i] << " ";
    // }
    // s1 << std::endl;

    // s2 << cnt << std::endl;
    // for (int i = 0; i < cnt; ++i) {
    //     s2 << (*hostBsrColInd)[i] << " ";
    // }
    // s2 << std::endl;

    return cnt;
}

int readAndFillBSRMatrix(int mb, int nb, int blockDim, int** hostBsrRowPtr, int** hostBsrColInd, float** hostBsrVal, float minVal=-1, float maxVal=1) {
    std::string bd = std::to_string(blockDim);
    std::fstream s1("bsr_" + bd + "_indptr.txt", std::ios::in);
    std::fstream s2("bsr_" + bd + "_indices.txt", std::ios::in);

    int xx;
    s1 >> xx;
    assert(mb + 1 == xx);
    *hostBsrRowPtr = (int*) malloc((mb + 1) * sizeof(int));
    for (int i = 0; i <= mb; ++i) {
        s1 >> (*hostBsrRowPtr)[i];
    }

    int nnzb;
    s2 >> nnzb;
    *hostBsrColInd = (int*) malloc(nnzb * sizeof(int));
    for (int i = 0; i < nnzb; ++i) {
        s2 >> (*hostBsrColInd)[i];
    }

    int num = nnzb * blockDim * blockDim;
    *hostBsrVal = (float*) malloc(num * sizeof(float));
    std::uniform_real_distribution<float> dist(minVal, maxVal);
    for (int i = 0; i < num; ++i) {
        (*hostBsrVal)[i] = dist(gen);
    }

    return nnzb;
}    

float* randomDenseMatrix(int n, int dim, float minVal=-1, float maxVal=1) {
    std::uniform_real_distribution<float> dist(minVal, maxVal);
    int sz = n * dim;
    float* ptr = (float*) malloc(sz * sizeof(float));
    for (int i = 0; i < sz; ++i) {
        ptr[i] = dist(gen);
    }
    return ptr;
}

#define CLEANUP(s) \
do { \
    printf("%s\n", s); \
    if (hostBsrRowPtr) free(hostBsrRowPtr); \
    if (hostBsrColInd) free(hostBsrColInd); \
    if (hostBsrVal) free(hostBsrVal); \
    if (yHostPtr) free(yHostPtr); \
    if (zHostPtr) free(zHostPtr); \
    if (bsrRowPtr) cudaFree(bsrRowPtr); \
    if (bsrColInd) cudaFree(bsrColInd); \
    if (bsrVal) cudaFree(bsrVal); \
    if (y) cudaFree(y); \
    if (y0) cudaFree(y0); \
    if (z) cudaFree(z); \
    if (z0) cudaFree(z0); \
    cudaDeviceReset(); \
    fflush(stdout); \
} while (0)

#define HANDLE_ERROR( err ) \
if (err != cudaSuccess) { \
    printf("%s in %s at linedd %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
    CLEANUP("cuda error occurred"); \
    exit(-1); \
}

#define HANDLE_CUSPARSE_ERROR( err, s ) \
if (err != CUSPARSE_STATUS_SUCCESS) { \
    CLEANUP(s); \
    exit(-1); \
}

int main(int argc, char* argv[]) {
    float p = std::stof(argv[1]);
    int blockDim = std::stoi(argv[2]); 
    int dim = std::stoi(argv[3]);
    printf("p = %f blockDim = %d dim = %d\n", p, blockDim, dim);

    cusparseHandle_t handle = 0;
    cusparseMatDescr_t descr = 0;

    int m = 2 << 16;
    int n = m;
    int mb = (m + blockDim - 1) / blockDim;
    int nb = (n + blockDim - 1) / blockDim;
    assert(mb * blockDim == m && nb * blockDim == n);
    int nnzb = 0;
    float fzero = 0.0;
    float fone = 1.0;

    int* hostBsrRowPtr = 0;
    int* hostBsrColInd = 0;
    float* hostBsrVal = 0;
    int* bsrRowPtr = 0;
    int* bsrColInd = 0;
    float* bsrVal = 0;

    float* yHostPtr = 0;
    float* y = 0;
    float* y0 = 0;
    float* zHostPtr = 0;
    float* z = 0;
    float* z0 = 0;

    printf("generate random BSR matrix\n");

    nnzb = randomBSRMatrix(mb, nb, blockDim, p, &hostBsrRowPtr, &hostBsrColInd, &hostBsrVal);
    // nnzb = readAndFillBSRMatrix(mb, nb, blockDim, &hostBsrRowPtr, &hostBsrColInd, &hostBsrVal);

    printf("density of BSR matrix is %f\n", (nnzb * 1.0) / (mb * nb));

    printf("gpu memory malloc and memcpy...\n");

    HANDLE_ERROR( cudaMalloc((void**)&bsrRowPtr, (mb + 1) * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&bsrColInd, nnzb * sizeof(int)) );
    HANDLE_ERROR( cudaMalloc((void**)&bsrVal, nnzb * blockDim * blockDim * sizeof(float)) );

    HANDLE_ERROR( cudaMemcpy(bsrRowPtr, hostBsrRowPtr, (size_t)((mb + 1) * sizeof(int)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(bsrColInd, hostBsrColInd, (size_t)(nnzb * sizeof(int)), cudaMemcpyHostToDevice) );
    HANDLE_ERROR( cudaMemcpy(bsrVal, hostBsrVal, (size_t)(nnzb * blockDim * blockDim * sizeof(float)), cudaMemcpyHostToDevice) );
    
    HANDLE_CUSPARSE_ERROR( cusparseCreate(&handle), "CUSPARSE Library initialization failed" );
    
    HANDLE_CUSPARSE_ERROR( cusparseCreateMatDescr(&descr), "BSR Matrix descriptor initialization failed" );
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    printf("prepare y and z...\n");

    yHostPtr = randomDenseMatrix(n, dim);
    zHostPtr = (float*) malloc(m * dim * sizeof(float));

    HANDLE_ERROR( cudaMalloc((void**)&y, n * dim * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&z, m * dim * sizeof(float)) );

    HANDLE_ERROR( cudaMemcpy(y, yHostPtr, (size_t)(n * dim * sizeof(float)), cudaMemcpyHostToDevice) );

    printf("warm up...\n");
    HANDLE_ERROR( cudaMalloc((void**)&y0, n * dim * sizeof(float)) );
    HANDLE_ERROR( cudaMalloc((void**)&z0, m * dim * sizeof(float)) );
    HANDLE_ERROR( cudaMemset((void*)y0, 0, n * dim * sizeof(float)) );
    HANDLE_ERROR( cudaMemset((void*)z0, 0, m * dim * sizeof(float)) );
    int warnupRounds = 3;
    for (int i = 0; i < warnupRounds; ++i) {
        HANDLE_CUSPARSE_ERROR( cusparseSbsrmm(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_TRANSPOSE, mb, dim, nb, nnzb, &fone, descr, bsrVal,
                                              bsrRowPtr, bsrColInd, blockDim, y0, dim, &fzero, z0, m),
                               "warmup cusparseSbsrmm failed" );  
    }

    printf("cusparseSbsrmm...\n");
    float totalTime = 0;
    int rounds = 10;
    for (int i = 0; i < rounds; ++i) {
        HANDLE_ERROR( cudaMemset((void*)z, 0, m * dim * sizeof(float)) );

        float time;
        cudaEvent_t start, stop;
        HANDLE_ERROR( cudaEventCreate(&start) );
        HANDLE_ERROR( cudaEventCreate(&stop) );
        HANDLE_ERROR( cudaEventRecord(start, 0) );

        HANDLE_CUSPARSE_ERROR( cusparseSbsrmm(handle, CUSPARSE_DIRECTION_ROW, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_TRANSPOSE, mb, dim, nb, nnzb, &fone, descr, bsrVal,
                                              bsrRowPtr, bsrColInd, blockDim, y, dim, &fzero, z, m),
                               "cusparseSbsrmm failed" ); 

        HANDLE_ERROR( cudaEventRecord(stop, 0) );
        HANDLE_ERROR( cudaEventSynchronize(stop) );
        HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
        printf("[%d] bsrmm cost time:  %3.10f ms \n", i, time);   
        totalTime += time;
    }
    printf("average bsrmm cost time:  %3.10f ms \n", totalTime / rounds);   

    HANDLE_ERROR( cudaMemcpy(zHostPtr, z, (size_t)(m * dim * sizeof(float)), cudaMemcpyDeviceToHost) );

    HANDLE_CUSPARSE_ERROR( cusparseDestroyMatDescr(descr), "Matrix descriptor destruction failed" );
    descr = 0;
    HANDLE_CUSPARSE_ERROR( cusparseDestroy(handle), "CUSPARSE Library release of resources failed" );
    handle = 0;

    CLEANUP("end");

    return 0;
}