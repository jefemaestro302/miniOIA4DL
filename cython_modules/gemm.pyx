import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gemm_blocked_hpc(float[:, ::1] matriz_pesos,      # matriz_A
                     float[:, ::1] matriz_columnas,   # matriz_B
                     float[:, ::1] matriz_salida,     # matriz_C
                     int tam_bloque_filas,            # mc
                     int tam_bloque_cols,             # nc
                     int tam_bloque_comun,            # kc
                     int tam_micro_filas,             # mr
                     int tam_micro_cols):             # nr
    
    # Dimensiones totales
    cdef int total_filas = matriz_pesos.shape[0]      # M
    cdef int total_cols = matriz_columnas.shape[1]    # N
    cdef int total_comun = matriz_pesos.shape[1]      # K
    
    # Índices para recorrer las matrices
    cdef int i, j, k, ii, jj, l, iii, jjj 

    
    # Tamaños de los bloques que estamos procesando en cada momento
    cdef int filas_bloque, cols_bloque, comun_bloque
    cdef int filas_micro, cols_micro

    
    # troceo L3
    for j in range(0, total_cols, tam_bloque_cols):
        if (j + tam_bloque_cols <= total_cols):
            cols_bloque = tam_bloque_cols
        else:
            cols_bloque = total_cols -j

        for k in range(0, total_comun, tam_bloque_comun):
            if (k + tam_bloque_comun <= total_comun):
                comun_bloque = tam_bloque_comun
            else:
                comun_bloque = total_comun - k

            for i in range (0, total_filas, tam_bloque_filas):
                if(i + tam_bloque_filas <= total_filas):
                    filas_bloque = tam_bloque_filas
                else: 
                    filas_bloque = total_filas -i

                # troceo L2 (Micro-bloques)
                for ii in range(0, filas_bloque, tam_micro_filas):
                    if (ii + tam_micro_filas <= filas_bloque):
                        filas_micro = tam_micro_filas
                    else:
                        filas_micro = filas_bloque - ii

                    for jj in range(0, cols_bloque, tam_micro_cols):
                        if (jj + tam_micro_cols <= cols_bloque):
                            cols_micro = tam_micro_cols
                        else:
                            cols_micro = cols_bloque - jj

                        # --- PASO FINAL: El cálculo real ---
                        for l in range(comun_bloque):
                            for iii in range(filas_micro):
                                for jjj in range (cols_micro):
                                    matriz_salida[i+ii+iii, j+jj+jjj] += matriz_pesos[i+ii+iii,k+l] * matriz_columnas[k+l,j+jj+jjj]


                                

    
    return np.asarray(matriz_salida)



