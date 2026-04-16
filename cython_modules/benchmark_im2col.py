
import numpy as np
import time
import sys
import os

# Add parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cython_modules.im2col import im2col_forward_cython
from cython_modules.competition import im2col_parallel

def benchmark():
    batch_size = 64
    channels = 64
    h, w = 32, 32
    kh, kw = 3, 3
    stride = 1
    padding = 1
    
    input_tensor = np.random.randn(batch_size, channels, h, w).astype(np.float32)
    
    print(f"Benchmarking im2col for {input_tensor.shape} kernel {kh}x{kw}")
    
    # Warmup
    _ = im2col_forward_cython(input_tensor, kh, kw, stride, padding)
    _ = im2col_parallel(input_tensor, kh, kw, stride, padding)
    
    start = time.time()
    for _ in range(10):
        res1 = im2col_forward_cython(input_tensor, kh, kw, stride, padding)
    end = time.time()
    print(f"im2col_forward_cython (serial): {(end-start)/10:.4f}s")
    
    start = time.time()
    for _ in range(10):
        res2 = im2col_parallel(input_tensor, kh, kw, stride, padding)
    end = time.time()
    print(f"im2col_parallel (parallel): {(end-start)/10:.4f}s")
    
    if np.allclose(res1, res2):
        print("Results match!")
    else:
        print("Results DO NOT match!")

if __name__ == "__main__":
    benchmark()
