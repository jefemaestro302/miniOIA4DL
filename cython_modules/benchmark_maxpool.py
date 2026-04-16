
import numpy as np
import time
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cython_modules.competition import maxpool2d_parallel

def benchmark_maxpool():
    # Dimensiones de la capa 1 de OIANet (donde MaxPool fue lento: 0.0259s)
    B, C, H, W = 64, 32, 32, 32
    kh, kw = 2, 2
    stride = 2
    
    input_tensor = np.random.randn(B, C, H, W).astype(np.float32)
    
    print(f"Benchmarking MaxPool for {input_tensor.shape} kernel {kh}x{kw}")
    
    # 1. NumPy version
    from numpy.lib.stride_tricks import sliding_window_view
    start = time.time()
    for _ in range(100):
        windows = sliding_window_view(input_tensor, (kh, kw), axis=(2, 3))
        windows = windows[:, :, ::stride, ::stride]
        res_numpy = np.max(windows, axis=(4, 5))
    end = time.time()
    print(f"NumPy sliding_window_view: {(end-start)/100:.6f}s")
    
    # 2. Cython version
    # Warmup
    _ = maxpool2d_parallel(input_tensor, kh, stride)
    start = time.time()
    for _ in range(100):
        res_cython = maxpool2d_parallel(input_tensor, kh, stride)
    end = time.time()
    print(f"Cython maxpool2d_parallel: {(end-start)/100:.6f}s")
    
    if np.allclose(res_numpy, res_cython):
        print("Results match!")
    else:
        print("Results DO NOT match!")

if __name__ == "__main__":
    benchmark_maxpool()
