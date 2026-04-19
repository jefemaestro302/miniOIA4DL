import numpy as np

OPTIMIZED_MATMUL = True

#PISTA: es esta la mejor forma de hacer una matmul?
def matmul_biasses(A, B, C, bias, optimized=None):
    if optimized is None:
        optimized = OPTIMIZED_MATMUL
        
    if not optimized:
        
        m, p, n = A.shape[0], A.shape[1], B.shape[1]
        if C is None:
            C = np.zeros((m, n), dtype=np.float32)
        for i in range(m):
            for j in range(n):
                for k in range(p):
                    C[i][j] += A[i][k] * B[k][j]
                C[i][j] += bias[j]
        return C

    # IMPLEMENTACIÓN OPTIMIZADA (Nivel Básico: Vectorización con NumPy)
    # Utilizamos np.dot para realizar el producto de matrices de forma vectorizada
    # y sumamos el bias de forma eficiente aprovechando el broadcasting de NumPy.
    return np.dot(A, B) + bias
    # // FIN BLOQUE GENERADO CON IA

