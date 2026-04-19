from cython.parallel import prange
cimport cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def im2col_parallel(float[:,:,:,:] input_tensor, int kernel_height, int kernel_width, int stride, int padding):
    cdef int batch_size = input_tensor.shape[0]
    cdef int channels = input_tensor.shape[1]
    cdef int in_height = input_tensor.shape[2]
    cdef int in_width = input_tensor.shape[3]

    cdef int out_heigth = (in_height + 2 * padding - kernel_height) // stride + 1
    cdef int out_width = (in_width + 2 * padding - kernel_width) // stride + 1

    cdef int filas = channels * kernel_height * kernel_width
    cdef int columnas = batch_size * out_heigth * out_width
    
    cdef float[:, ::1] result = np.zeros((filas, columnas), dtype=np.float32)

    cdef int total_iters = batch_size * channels
    cdef int it, n, c, fy, fx, oh, ow
    cdef int h_in, w_in, idx_row, idx_col_base, idx_col

    # // INICIO BLOQUE GENERADO CON IA
    for it in prange(total_iters, nogil=True):
        n = it // channels
        c = it % channels
        
        idx_col_base = n * (out_heigth * out_width)
        
        for fy in range(kernel_height):
            for fx in range(kernel_width):
                idx_row = c * (kernel_height * kernel_width) + fy * kernel_width + fx
                for oh in range(out_heigth):
                    h_in = oh * stride + fy - padding
                    if h_in >= 0 and h_in < in_height:
                        for ow in range(out_width):
                            w_in = ow * stride + fx - padding
                            if w_in >= 0 and w_in < in_width:
                                idx_col = idx_col_base + oh * out_width + ow
                                result[idx_row, idx_col] = input_tensor[n, c, h_in, w_in]
    # // FIN BLOQUE GENERADO CON IA

    return np.asarray(result)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gemm_parallel_blocked(float[:, ::1] A, float[:, ::1] B, float[:, ::1] C,
                          int mc, int nc, int kc, int mr, int nr):
    cdef int M = A.shape[0]
    cdef int K = A.shape[1]
    cdef int N = B.shape[1]

    cdef int n_i = (M + mc - 1) // mc
    cdef int n_k = (K + kc - 1) // kc
    cdef int n_j = (N + nc - 1) // nc

    cdef int ib, kb, jb, i, j, k, i_limit, j_limit, k_limit
    cdef int ii, jj, l, iii, jjj, ii_limit, jj_limit
    cdef int n_ii, n_jj, ii_idx, jj_idx

    # // INICIO BLOQUE GENERADO CON IA
    for ib in prange(n_i, nogil=True):
        i = ib * mc
        if i + mc < M:
            i_limit = i + mc
        else:
            i_limit = M
            
        for kb in range(n_k):
            k = kb * kc
            if k + kc < K:
                k_limit = k + kc
            else:
                k_limit = K
            
            for jb in range(n_j):
                j = jb * nc
                if j + nc < N:
                    j_limit = j + nc
                else:
                    j_limit = N
                
                n_ii = (i_limit - i + mr - 1) // mr
                for ii_idx in range(n_ii):
                    ii = i + ii_idx * mr
                    if ii + mr < i_limit:
                        ii_limit = ii + mr
                    else:
                        ii_limit = i_limit
                    
                    n_jj = (j_limit - j + nr - 1) // nr
                    for jj_idx in range(n_jj):
                        jj = j + jj_idx * nr
                        if jj + nr < j_limit:
                            jj_limit = jj + nr
                        else:
                            jj_limit = j_limit
                        
                        for l in range(k, k_limit):
                            for iii in range(ii, ii_limit):
                                for jjj in range(jj, jj_limit):
                                    C[iii, jjj] = C[iii, jjj] + A[iii, l] * B[l, jjj]
    # // FIN BLOQUE GENERADO CON IA

    return np.asarray(C)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def maxpool2d_parallel(float[:,:,:,:] input_tensor, int kernel_size, int stride):
    cdef int batch_size = input_tensor.shape[0]
    cdef int channels = input_tensor.shape[1]
    cdef int in_height = input_tensor.shape[2]
    cdef int in_width = input_tensor.shape[3]
    
    cdef int out_heigth = (in_height - kernel_size) // stride + 1
    cdef int out_width = (in_width - kernel_size) // stride + 1
    
    cdef float[:,:,:,:] salida = np.zeros((batch_size, channels, out_heigth, out_width), dtype=np.float32)
    
    cdef int total_iters = batch_size * channels
    cdef int it, b, c, i, j, fy, fx
    cdef float max_val, valor_actual
    
    # // INICIO BLOQUE GENERADO CON IA
    for it in prange(total_iters, nogil=True):
        b = it // channels
        c = it % channels
        
        for i in range(out_heigth):
            for j in range(out_width):
                max_val = -1e30 
                for fy in range(kernel_size):
                    for fx in range(kernel_size):
                        valor_actual = input_tensor[b, c, i * stride + fy, j * stride + fx]
                        if valor_actual > max_val:
                            max_val = valor_actual
                salida[b, c, i, j] = max_val
    # // FIN BLOQUE GENERADO CON IA
                
    return np.asarray(salida)
