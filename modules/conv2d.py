from modules.layer import Layer
from modules.utils import *
from cython_modules.im2col import im2col_forward_cython
from cython_modules.gemm import gemm_blocked_hpc
from cython_modules.competition import im2col_parallel, gemm_parallel_blocked

import numpy as np

class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, conv_algo=0, weight_init="he"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # MODIFICAR: Añadir nuevo if-else para otros algoritmos de convolución
        if conv_algo == 0:
            self.mode = 'direct' 
        elif conv_algo == 1:
            self.mode = 'im2col_basic' # Nivel Básico (NumPy)
        elif conv_algo == 2:
            self.mode = 'im2col_cython' # Nivel Medio (Cython)
        elif conv_algo == 3:
            self.mode = 'gemm_blocked' # Nivel Avanzado (Blocked GEMM)
        elif conv_algo == 4:
            self.mode = 'competition' # Nivel Competitivo (Parallel + Blocked)
        else:
            print(f"Algoritmo {conv_algo} no soportado aún")
            self.mode = 'direct' 

        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size

        if weight_init == "he":
            std = np.sqrt(2.0 / fan_in)
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "xavier":
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.kernels = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * std
        elif weight_init == "custom":
            self.kernels = np.zeros((out_channels, in_channels, kernel_size, kernel_size), dtype=np.float32)
        else:
            self.kernels = np.random.uniform(-0.1, 0.1, 
                          (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        

        self.biases = np.zeros(out_channels, dtype=np.float32)

        # PISTA: Y estos valores para qué las podemos utilizar?
        # Si los usas, no olvides utilizar el modelo explicado en teoría que maximiza la caché
        self.mc = 480
        self.nc = 3072
        self.kc = 384
        self.mr = 32
        self.nr = 12
        self.Ac = np.empty((self.mc, self.kc), dtype=np.float32)
        self.Bc = np.empty((self.kc, self.nc), dtype=np.float32)


    def get_weights(self):
        return {'kernels': self.kernels, 'biases': self.biases}

    def set_weights(self, weights):
        self.kernels = weights['kernels']
        self.biases = weights['biases']
    
    def forward(self, input, training=True):
        self.input = input
        # PISTA: Usar estos if-else si implementas más algoritmos de convolución
        if self.mode == 'direct':
            return self._forward_direct(input)
        elif self.mode == 'im2col_basic':
            return self._forward_im2col_basic(input)
        elif self.mode == 'im2col_cython':
            return self._forward_im2col_cython(input)
        elif self.mode == 'gemm_blocked':
            return self._forward_gemm_blocked(input)
        elif self.mode == 'competition':
            return self._forward_competition(input)
        else:
            raise ValueError(f"Mode {self.mode} not supported")

    def _forward_im2col_basic(self, input):
        batch_size, channels, in_height, in_width = input.shape
        kernel_height, kernel_width = self.kernel_size, self.kernel_size
        out_heigth = (in_height + 2 * self.padding - kernel_height) //self.stride + 1
        out_width = (in_width + 2 * self.padding - kernel_width) // self.stride + 1
        
        if self.padding > 0:
            input_padded = np.pad(input, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)),mode='constant').astype(np.float32)
        else:
            input_padded = input
        
        # en vez de iterar con bucles for cada región, usamos Numpy para crear una vista de cada ventana de tamaño kernelxkernel
        # explorando los ejes de altura y anchura (kernel_heigth, kernel_width)
        from numpy.lib.stride_tricks import sliding_window_view
        ventanas = sliding_window_view(input_padded, (kernel_height,kernel_width),axis=(2,3))
        # el paso anterior explora pixel a pixel; esto puede no coincidir con nuestro stride, asi que nos saltamos las ventans inutiles
        ventanas = ventanas[: , : , ::self.stride, ::self.stride, :, :]

        #reordenamos la matriz para que los datos de cada ventana estén juntos 
        # reordenamos la matriz para que los datos de cada ventana estén juntos 
        # canales, alto_v, ancho_v, batch, alto_out, ancho_out
        ventanas = ventanas.transpose(1,4,5,0,2,3)
        columnas_entrada = ventanas.reshape(channels * kernel_height * kernel_width, -1)
        #sustituimos convolucion por multiplicacion
        pesos = self.kernels.reshape(self.out_channels, -1)
        salida_aplanada = np.dot(pesos, columnas_entrada) + self.biases.reshape(-1,1)

        #reformateamos a formato tensor
        salida= salida_aplanada.reshape(self.out_channels, batch_size, out_heigth, out_width)
        return salida.transpose(1, 0, 2, 3)


        

    def _forward_im2col_cython(self, input):
        # implementacion cython de i2col
        batch_size, channels, in_height, in_width = input.shape
        kernel_height, kernel_width = self.kernel_size, self.kernel_size
        
        # Calcular dimensiones de salida
        out_heigth = (in_height + 2 * self.padding - kernel_height) //self.stride + 1
        out_width = (in_width + 2 * self.padding - kernel_width) // self.stride + 1

        # realizamos las transformaciones con el modulo cython
        columnas_entrada = im2col_forward_cython(input, kernel_height, kernel_width, self.stride, self.padding)
        
    
        pesos = self.kernels.reshape(self.out_channels, -1)
        
        #aplanamos la salida y realizamos la multiplicacion
        output_flat = np.dot(pesos, columnas_entrada) + self.biases.reshape(-1, 1)
        
        #volvemos al formato tensor
        salida = output_flat.reshape(self.out_channels, batch_size, out_heigth, out_width)
        return salida.transpose(1, 0, 2, 3)

    def _forward_gemm_blocked(self, input):
        # // INICIO BLOQUE GENERADO CON IA
        # NIVEL ALTO: im2col + Custom Blocked GEMM
        batch_size = input.shape[0]
        k_h, k_w = self.kernel_size, self.kernel_size
        
        # 1. Transformar entrada a matriz de columnas usando Cython
        input_cols = im2col_forward_cython(input, k_h, k_w, self.stride, self.padding)
        
        # 2. Preparar matrices para GEMM
        A = self.kernels.reshape(self.out_channels, -1).astype(np.float32) # (M, K)
        B = input_cols.astype(np.float32) # (K, N)
        M, K = A.shape
        N = B.shape[1]
        
        # Pre-preparar matriz de salida
        C = np.zeros((M, N), dtype=np.float32)
        
        # 3. Llamar al motor GEMM avanzado
        output_flat = gemm_blocked_hpc(A, B, C, self.mc, self.nc, self.kc, self.mr, self.nr)
        
        # 4. Sumar biases (broadcast)
        output_flat += self.biases.reshape(-1, 1)
        
        # 5. Re-formatear a tensor de salida (N, OutC, OutH, OutW)
        out_h = (input.shape[2] + 2 * self.padding - k_h) // self.stride + 1
        out_w = (input.shape[3] + 2 * self.padding - k_w) // self.stride + 1
        output = output_flat.reshape(self.out_channels, batch_size, out_h, out_w)
        return output.transpose(1, 0, 2, 3)
        # // FIN BLOQUE GENERADO CON IA

    def _forward_competition(self, input):
        # // INICIO BLOQUE GENERADO CON IA
        # NIVEL COMPETITIVO: Parallel im2col (Cython) + np.dot (BLAS)
        batch_size = input.shape[0]
        k_h, k_w = self.kernel_size, self.kernel_size
        
        # 1. im2col paralelo (altamente optimizado en Cython, evita np.pad)
        # Aseguramos float32 una sola vez
        if input.dtype != np.float32:
            input_f32 = input.astype(np.float32)
        else:
            input_f32 = input

        input_cols = im2col_parallel(input_f32, k_h, k_w, self.stride, self.padding)
        
        # 2. GEMM usando np.dot (BLAS)
        weights_reshaped = self.kernels.reshape(self.out_channels, -1)
        output_flat = np.dot(weights_reshaped, input_cols) + self.biases.reshape(-1, 1)
        
        # 3. Reshape y transpose final
        out_h = (input.shape[2] + 2 * self.padding - k_h) // self.stride + 1
        out_w = (input.shape[3] + 2 * self.padding - k_w) // self.stride + 1
        output = output_flat.reshape(self.out_channels, batch_size, out_h, out_w)
        return output.transpose(1, 0, 2, 3)
        # // FIN BLOQUE GENERADO CON IA

    def backward(self, grad_output, learning_rate):
        # ESTO NO ES NECESARIO YA QUE NO VAIS A HACER BACKPROPAGATION
        if self.mode == 'direct':
            return self._backward_direct(grad_output, learning_rate)
        else:
            raise ValueError("Mode must be 'direct' or 'im2col'")

    # --- DIRECT IMPLEMENTATION ---

    def _forward_direct(self, input):
        batch_size, _, in_h, in_w = input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input = np.pad(input,
                           ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                           mode='constant').astype(np.float32)

        out_h = (input.shape[2] - k_h) // self.stride + 1
        out_w = (input.shape[3] - k_w) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_h, out_w), dtype=np.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            region = input[b, in_c,
                                           i * self.stride:i * self.stride + k_h,
                                           j * self.stride:j * self.stride + k_w]
                            output[b, out_c, i, j] += np.sum(region * self.kernels[out_c, in_c])
                output[b, out_c] += self.biases[out_c]

        return output

    def _backward_direct(self, grad_output, learning_rate):
        batch_size, _, out_h, out_w = grad_output.shape
        _, _, in_h, in_w = self.input.shape
        k_h, k_w = self.kernel_size, self.kernel_size

        if self.padding > 0:
            input_padded = np.pad(self.input,
                                  ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                                  mode='constant').astype(np.float32)
        else:
            input_padded = self.input

        grad_input_padded = np.zeros_like(input_padded, dtype=np.float32)
        grad_kernels = np.zeros_like(self.kernels, dtype=np.float32)
        grad_biases = np.zeros_like(self.biases, dtype=np.float32)

        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for in_c in range(self.in_channels):
                    for i in range(out_h):
                        for j in range(out_w):
                            r = i * self.stride
                            c = j * self.stride
                            region = input_padded[b, in_c, r:r + k_h, c:c + k_w]
                            grad_kernels[out_c, in_c] += grad_output[b, out_c, i, j] * region
                            grad_input_padded[b, in_c, r:r + k_h, c:c + k_w] += self.kernels[out_c, in_c] * grad_output[b, out_c, i, j]
                grad_biases[out_c] += np.sum(grad_output[b, out_c])

        if self.padding > 0:
            grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            grad_input = grad_input_padded

        self.kernels -= learning_rate * grad_kernels
        self.biases -= learning_rate * grad_biases

        return grad_input

    # PISTA: Se te ocurren otros algoritmos de convolución?
