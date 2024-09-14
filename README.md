# spMM-benchmark
These are sample programs I've written to test the formance of Triton kernels. You may find more details about each program below!

## Sample Programs
`spMM_CSR.py`  - A naive Triton kernel performing spMM using Gustavson's algorithm. Also includes a handler and benchmarking methods 

`spMM_CSR_blocked.py` - Same as above except the kernel parallelizes over tiles instead of rows 

`cuSPARSE_spMM_bsr.cu` - A simple Python binding for cuSPARSE's cusparseSbsrmm function for blocked spMM 

`spMM_test.py` - A function that tests all of these alongside cuPy's spMM implementation. The `main()` function takes as input `dim` 
(row and col dimensions for input matrices), `block_size` size of tiles for blocked implementations and `density`, which dictates input matrix sparsity 

`Makefile` - running `make` runns multiple experiments benchmarking all implementations. You may change the parameters to the experiments by changing the values inside this file

## Dependencies
This project requires you to have Python, cuPy, PyTorch, SciPy and the CUDA Toolkit installed.

## Running
To run all experiments with default values:
```
make
```

To compile the cuSPARSE bsr spMM implementation:
```
nvcc -o libcuSPARSE_spMM_bsr.so cuSPARSE_spMM_bsr.cu -shared -lcusparse -lcublas -lcudart -Xcompiler -fPIC
```
