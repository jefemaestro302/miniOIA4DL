import numpy as np
import time
from modules.conv2d import Conv2D

def verify_and_benchmark():
    # Parámetros (un poco más grandes para notar la diferencia)
    batch_size = 8
    in_channels = 16
    out_channels = 32
    img_size = 32
    kernel_size = 3
    stride = 1
    padding = 1

    input_data = np.random.randn(batch_size, in_channels, img_size, img_size).astype(np.float32)

    # 1. Modo Directo (conv_algo=0)
    conv_direct = Conv2D(in_channels, out_channels, kernel_size, stride, padding, conv_algo=0)
    
    start = time.time()
    out_direct = conv_direct.forward(input_data)
    time_direct = time.time() - start
    print(f"Directo (0): {time_direct:.4f} segundos")

    # 2. Modo im2col Básico (conv_algo=1)
    conv_basic = Conv2D(in_channels, out_channels, kernel_size, stride, padding, conv_algo=1)
    conv_basic.kernels = conv_direct.kernels.copy()
    conv_basic.biases = conv_direct.biases.copy()

    start = time.time()
    out_basic = conv_basic.forward(input_data)
    time_basic = time.time() - start
    print(f"Básico (1) :  {time_basic:.4f} segundos")

    # 3. Modo im2col Medio/Cython (conv_algo=2)
    conv_cython = Conv2D(in_channels, out_channels, kernel_size, stride, padding, conv_algo=2)
    conv_cython.kernels = conv_direct.kernels.copy()
    conv_cython.biases = conv_direct.biases.copy()

    start = time.time()
    out_cython = conv_cython.forward(input_data)
    time_cython = time.time() - start
    print(f"Medio (2)  :  {time_cython:.4f} segundos")

    # Verificación
    if np.allclose(out_direct, out_basic, atol=1e-5) and np.allclose(out_direct, out_cython, atol=1e-5):
        print("\n✅ ¡ÉXITO! Los 3 resultados coinciden perfectamente.")
        print(f"🚀 Speedup Directo -> Básico: {time_direct / time_basic:.1f}x")
        print(f"🚀 Speedup Básico -> Medio : {time_basic / time_cython:.1f}x")
        print(f"🚀 Speedup Directo -> Medio : {time_direct / time_cython:.1f}x")
    else:
        print("❌ ERROR: Los resultados no coinciden.")
        # Debugging opcional si falla
        # print(np.abs(out_direct - out_im2col).max())

if __name__ == "__main__":
    verify_and_benchmark()
