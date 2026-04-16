import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def im2col_forward_cython(np.ndarray[np.float32_t, ndim=4] input_tensor, 
                          int kernel_h, int kernel_w, 
                          int stride, int padding):
    """
    Transforma un tensor de entrada (N, C, H, W) en una matriz de columnas (GEMM-ready).
    """
    cdef int batch_size = input_tensor.shape[0]
    cdef int channels = input_tensor.shape[1]
    cdef int in_h = input_tensor.shape[2]
    cdef int in_w = input_tensor.shape[3]

    cdef int out_h = (in_h + 2 * padding - kernel_h) // stride + 1
    cdef int out_w = (in_w + 2 * padding - kernel_w) // stride + 1

    # Gestión de Padding
    cdef np.ndarray[np.float32_t, ndim=4] input_padded
    if padding > 0:
        input_padded = np.pad(input_tensor, 
                             ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                             mode='constant').astype(np.float32)
    else:
        input_padded = input_tensor

    # Reserva de memoria para la matriz de columnas
    cdef int rows = channels * kernel_h * kernel_w
    cdef int cols = batch_size * out_h * out_w
    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros((rows, cols), dtype=np.float32)

    cdef int batch_idx, out_y, out_x, channel_idx, filter_y, filter_x
    cdef int col_idx, row_idx

    for batch_idx in range(batch_size):
        for out_y in range(out_h):
            for out_x in range(out_w):
                # Índice de la columna actual (posición de la ventana en el batch)
                col_idx = batch_idx * (out_h * out_w) + out_y * out_w + out_x
                
                row_idx = 0
                for channel_idx in range(channels):
                    for filter_y in range(kernel_h):
                        for filter_x in range(kernel_w):
                            # Copiar píxel de la imagen a la matriz de columnas
                            result[row_idx, col_idx] = input_padded[batch_idx, channel_idx, 
                                                                   out_y * stride + filter_y, 
                                                                   out_x * stride + filter_x]
                            row_idx += 1
    
    return result
    