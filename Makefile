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
BLOCK_SIZES = 32
DIMS = 2048
DENSITIES = 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21 0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32 0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43 0.44 0.45 0.46 0.47 0.48 0.49 0.5

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
	rm -f *.png
	rm -f *.txt