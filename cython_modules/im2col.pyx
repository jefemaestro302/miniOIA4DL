import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)

#creamos objetos C puros con Cython
def im2col_forward_cython(np.ndarray[np.float32_t, ndim=4] input_tensor, 
                          int kernel_height, int kernel_width, 
                          int stride, int padding):
    
    cdef int batch_size = input_tensor.shape[0]
    cdef int channels = input_tensor.shape[1]
    cdef int in_height = input_tensor.shape[2]
    cdef int in_width = input_tensor.shape[3]

    cdef int out_heigth = (in_height + 2 * padding - kernel_height) // stride + 1
    cdef int out_width = (in_width + 2 * padding - kernel_width) // stride + 1

    # comprobamos el padding antes de los bucles para evitar comprobarlo en cada vuelta 
    cdef np.ndarray[np.float32_t, ndim=4] input_padded
    if padding > 0:
        input_padded = np.pad(input_tensor, 
                             ((0, 0), (0, 0), (padding, padding), (padding, padding)), 
                             mode='constant').astype(np.float32)
    else:
        input_padded = input_tensor

    # reservamos la memoria de las matrices
    # las filas son funcion de la cantidad de canales del filtro * las dimensiones del kernel
    cdef int rows = channels * kernel_height * kernel_width
    #las columnas son funcions de la cantidad de imagenes 
    cdef int cols = batch_size * out_heigth * out_width
    #reserveamos la matriz bidimensional; filas -> canales * ancho alto kernel
    #columnas -> batch * alto salida * largo salida 
    cdef np.ndarray[np.float32_t, ndim=2] result = np.zeros((rows, cols), dtype=np.float32)

    cdef int batch_idx, out_y, out_x, channel_idx, filter_y, filter_x
    cdef int column_idx, row_idx
    
    # identificamos que imagen del batch somos
    for batch_idx in range(batch_size):
        #identificamos los margenes de la ventana en el eje y
        for out_y in range(out_heigth):
            #recorremos la imagen por filas, revisando todas las columnas para cada posicion
            for out_x in range(out_width):
                # igual, pero del eje x
                column_idx = batch_idx * (out_heigth * out_width) + out_y * out_width + out_x
                

                
                row_idx = 0
                #  estamos dentro de una de las posibles ventanas, por lo que recorrermos cada canal (RGB) del filtro 
                for channel_idx in range(channels):
                    for filter_y in range(kernel_height):
                        for filter_x in range(kernel_width):
                            # copiamos el valor del pixel particular de la matriz tridimensional en la columna
                            result[row_idx, column_idx] = input_padded[batch_idx, channel_idx, 
                                                                   out_y * stride + filter_y, 
                                                                   out_x * stride + filter_x]
                            row_idx += 1
    
    return result
    