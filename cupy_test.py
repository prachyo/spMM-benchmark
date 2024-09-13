import cupy as cp
print(cp.cuda.runtime.getDeviceCount())  # Should print the number of available CUDA devices