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

# Values for the loops
BLOCK_SIZES = 32 64 128
DIMS = 1024 2048 4096
DENSITIES = 0.01 0.1 0.5

# Rule for running the Python script
run_python:
	@for block_size in $(BLOCK_SIZES); do \
		for dim in $(DIMS); do \
			for density in $(DENSITIES); do \
				python3 spMM_test.py $$dim $$density $$block_size; \
			done; \
		done; \
	done

# Clean up
clean:
	rm -f $(CUDA_LIB)