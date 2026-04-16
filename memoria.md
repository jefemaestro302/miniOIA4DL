# Memoria de Optimización - miniOIA4DL

## 1. Estado Inicial
Implementación "directa" de la capa convolucional ineficiente.
- **IPS global:** ~0.85

## 2. Optimización 1: Algoritmo im2col (Nivel Básico)
- **Implementación:** `sliding_window_view` de NumPy + `np.dot`.
- **IPS global:** ~222

## 3. Optimización 2: im2col en Cython (Nivel Medio)
- **Implementación:** Módulo Cython serial para `im2col`.
- **IPS global:** ~90 (limitado por overhead de padding externo y falta de paralelismo)

## 4. Optimización 3: Nivel Competitivo (Paralelización Total)

### Identificación del Objetivo
Eliminar cuellos de botella en `Conv2D` y `MaxPool2D` mediante paralelismo masivo y gestión de memoria eficiente (zero-copy padding).

### Implementación
Se desarrolló el módulo `cython_modules/competition.pyx` incluyendo:
- **`im2col_parallel`**: Paralelización sobre `batch * channels`. Gestión manual de padding (sin `np.pad`).
- **`maxpool2d_parallel`**: Reemplazo de la lógica de ventanas de NumPy por un kernel paralelo en Cython (~39x más rápido que NumPy).
- **Rendimiento Final:** **275.79 IPS** (batch size 64).
- **Speedup Global:** **~324x** respecto al modo directo.

### Estado Actual
El framework es ahora capaz de realizar inferencia a alta velocidad, superando ampliamente las implementaciones estándar basadas en NumPy genérico.

## 5. Próximos Pasos
- Explorar cuantización (int8) para reducir aún más los tiempos de acceso a memoria.
- Optimizar BatchNorm2D mediante fusión de kernels con la capa ReLU.
