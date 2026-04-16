# Guía de Optimizaciones HPC - miniOIA4DL

Esta guía detalla la estrategia y el razonamiento técnico detrás de las optimizaciones realizadas para alcanzar el Nivel Competitivo en el framework.

## 1. ¿Por qué optimizar estas capas?

Tras realizar un perfilado (profiling) del modelo `OIANet`, se identificaron dos cuellos de botella críticos:

1.  **Conv2D (~80% del tiempo inicial):** La implementación directa usaba 6 bucles anidados en Python. Incluso la versión `im2col` en Cython serial era ineficiente debido al uso de `np.pad` (copias de memoria) y la falta de paralelismo en la transformación de la imagen a matriz.
2.  **MaxPool2D (~15% del tiempo tras optimizar Conv2D):** Aunque usaba `sliding_window_view` de NumPy, esta operación crea vistas complejas que penalizan el rendimiento en accesos posteriores. Al reducir el tiempo de la convolución, el pooling se convirtió en el nuevo limitador.

## 2. ¿Cómo lo hemos optimizado? (La Estrategia)

### A. La Transformación im2col Paralela
En lugar de depender de funciones genéricas de NumPy, implementamos un motor en Cython con las siguientes características:
-   **Eliminación de Padding Externo:** El padding se gestiona mediante lógica condicional (`if`) dentro de los bucles de Cython. Esto evita crear una copia de la imagen con bordes de ceros (`np.pad`), ahorrando tiempo y ancho de banda de memoria.
-   **Colapso de Dimensiones:** Paralelizamos sobre el producto de `Batch Size * Canales`. Esto asegura que todos los núcleos de la CPU estén ocupados, incluso en la primera capa de la red donde solo hay 3 canales.
-   **Escritura Contigua:** El orden de los bucles se diseñó para que la escritura en la matriz de resultados sea lo más secuencial posible, aprovechando la jerarquía de caché.

### B. GEMM: ¿Por qué usamos np.dot?
Aunque implementamos un GEMM bloqueado manualmente (`gemm_parallel_blocked`), los resultados mostraron que `np.dot` es superior.
-   **Razón:** `np.dot` enlaza con librerías como OpenBLAS o MKL, que utilizan **instrucciones SIMD (AVX/AVX-512)** y un ensamblador altamente optimizado para cada arquitectura. Nuestra implementación manual en C, aunque bloqueada para caché, no puede competir con el nivel de micro-optimización de una librería BLAS industrial.

### C. MaxPool2D Paralelo
Sustituimos la implementación de NumPy por un kernel de Cython que:
-   Paraleliza sobre `Batch * Canales`.
-   Realiza la búsqueda del máximo en un solo paso, evitando la creación de tensores intermedios o vistas de memoria complejas.

## 3. Comparativa de Rendimiento (OIANet)

| Algoritmo | Tiempo Batch (s) | IPS (Imágenes/seg) | Speedup Global |
| :--- | :--- | :--- | :--- |
| **Directo (0)** | ~23.68s | ~0.3 | 1x |
| **im2col Cython (2)** | ~0.09s | ~90 | ~300x |
| **Competitivo (4)** | **~0.04s** | **~181** | **~600x** |

*Nota: Mediciones realizadas con Batch Size = 64.*

## 4. Conclusión
La clave de la optimización HPC no es solo "hacerlo en C", sino entender el **flujo de datos**. Al minimizar las copias de memoria (sin padding externo) y maximizar el paralelismo en las capas de transformación, hemos logrado que el framework sea capaz de procesar imágenes a una velocidad competitiva para tareas de aprendizaje profundo.
