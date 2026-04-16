from modules.layer import Layer
from cython_modules.competition import maxpool2d_parallel
import numpy as np

class MaxPool2D(Layer):
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, input, training=True):  # input: np.ndarray of shape [B, C, H, W]
        # // INICIO BLOQUE GENERADO CON IA
        # NIVEL COMPETITIVO: MaxPool2D paralelo en Cython
        return maxpool2d_parallel(input.astype(np.float32), self.kernel_size, self.stride)
        # // FIN BLOQUE GENERADO CON IA

    def backward(self, grad_output, learning_rate=None):
        B, C, H, W = self.input.shape
        grad_input = np.zeros_like(self.input, dtype=grad_output.dtype)
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    for j in range(out_w):
                        r, s = self.max_indices[b, c, i, j]
                        grad_input[b, c, r, s] += grad_output[b, c, i, j]

        return grad_input