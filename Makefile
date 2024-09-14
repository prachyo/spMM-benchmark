# Makefile for compiling CUDA code and running a Python test

# Variables
CUDA_SOURCES = cuSPARSE_spMM_bsr.cu
CUDA_LIB = libcuSPARSE_spMM_bsr.so
PYTHON_SCRIPT = spMM_test.py

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -shared -lcusparse -lcublas -lcudart -Xcompiler -fPIC

# Default target
all: $(CUDA_LIB) run_python

# Rule for building the CUDA shared library
$(CUDA_LIB): $(CUDA_SOURCES)
	$(NVCC) -o $(CUDA_LIB) $(CUDA_SOURCES) $(NVCC_FLAGS)

# Rule for running the Python script
run_python:
	python3 $(PYTHON_SCRIPT)

# Clean up
clean:
	rm -f $(CUDA_LIB)