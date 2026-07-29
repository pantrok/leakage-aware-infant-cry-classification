# Resultados parciales de la corrida en Colab (sesión cortada por límite free)

**Estado: PARCIAL, de referencia — no es la corrida oficial.** La sesión
gratuita de Colab terminó por límite de tiempo antes de terminar el
grid completo. Esto se rescató directo del output embebido en la celda
de `leakage` de `notebooks/colab_run.ipynb` (Colab guarda el output en
el propio notebook, sincronizado a Drive) — nunca se alcanzó a escribir
en `results/leakage_results.csv`/`leakage_stats.csv` porque el código
que corrió esa sesión (previo al commit `0c95f21` de checkpoint/resume)
solo guardaba CSV al final de *todo* el grid, y la sesión murió a media
carga de `binary/E` sin ninguna excepción de Python de por medio (kill
directo, no hubo `finally` que disparar).

Por qué vale la pena guardarlo de todos modos: confirma con datos reales
que **la corrección de `_recording_id()` (M=73) y el wiring de GPU
(cuML + PyTorch) funcionan correctamente en Colab** — no son solo
teoría. Sirve como punto de referencia/sanity-check para comparar contra
la corrida completa y oficial que hará el colega en su máquina, no para
citar en el paper.

## Confirmaciones de esta corrida

```
[INFO] Backend de cómputo: cuML (GPU NVIDIA, vía RAPIDS) para SVM/kNN.
[INFO] PyTorch detecta GPU CUDA disponible para el MLP.
...
[INFO] GroupKFold: 73 grabaciones únicas, 31.1 segmentos/grabación promedio.
```

- GPU real confirmada (no CPU/sklearn) — el wiring de `src/gpu_classifiers.py`
  en `leakage_comparison.py` funciona.
- M = 73 grabaciones únicas confirmado otra vez, en un entorno distinto
  (Colab/Linux) al de la verificación original (Windows local).
- Multiclase: los 4 conjuntos (B, C, D, E) terminaron completos, ambas
  condiciones (S-1s, G-1s).
- Binario: B, C, D terminaron completos, ambas condiciones. E se cortó
  literalmente en "Cargando carpetas 1s_X..." — cero resultados de esa
  combinación.
- **No se generaron pruebas de significancia (Wilcoxon/Bonferroni) ni
  figuras** — eso requiere que las 8 combinaciones completen, y solo
  llegaron 7 de 8.

## Tiempos observados (referencia de costo real con GPU)

| Combinación | Tiempo |
|---|---:|
| multiclass / B | 1240.2 s (~21 min) |
| multiclass / C | 1237.2 s (~21 min) |
| multiclass / D | 1227.8 s (~20 min) |
| multiclass / E | 10782.1 s (~3 h) |
| binary / B | 1049.2 s (~17 min) |
| binary / C | 1089.8 s (~18 min) |
| binary / D | 1140.2 s (~19 min) |
| binary / E | cortada, sin completar |

Conjunto E (Fisher Vector, GMM por fold) sigue siendo, por mucho, el más
caro incluso con GPU — el cuello de botella ahí es la extracción de
características (CPU) y el ajuste del GMM, no el clasificador en sí.

## Tabla Kappa/Accuracy — MULTICLASE (completa, las 4 combinaciones)

### Conjunto B — comp+EEFGabor (sin FOSP)

| Condición | Clasificador | Kappa | IC95% | Acc |
|---|---|---:|---|---:|
| S-1s | SVM-RBF | 0.9133 ± 0.003 | [0.908, 0.917] | 0.9354 |
| S-1s | RandomForest | 0.9122 ± 0.002 | [0.909, 0.915] | 0.9346 |
| S-1s | SVM-polinomial | 0.9034 ± 0.003 | [0.900, 0.907] | 0.9280 |
| S-1s | MLP | 0.8812 ± 0.004 | [0.878, 0.888] | 0.9114 |
| S-1s | SVM-lineal | 0.8693 ± 0.003 | [0.865, 0.872] | 0.9025 |
| S-1s | kNN | 0.8454 ± 0.006 | [0.839, 0.854] | 0.8836 |
| G-1s | SVM-lineal | 0.6462 ± 0.002 | [0.644, 0.648] | 0.7332 |
| G-1s | MLP | 0.6249 ± 0.006 | [0.617, 0.632] | 0.7170 |
| G-1s | SVM-RBF | 0.6156 ± 0.001 | [0.614, 0.618] | 0.7101 |
| G-1s | RandomForest | 0.5920 ± 0.006 | [0.584, 0.599] | 0.6931 |
| G-1s | SVM-polinomial | 0.5787 ± 0.003 | [0.576, 0.584] | 0.6798 |
| G-1s | kNN | 0.5554 ± 0.004 | [0.551, 0.561] | 0.6576 |

### Conjunto C — FOSP+comp+EEFGabor (sin reducción)

| Condición | Clasificador | Kappa | IC95% | Acc |
|---|---|---:|---|---:|
| S-1s | RandomForest | 0.9077 ± 0.003 | [0.903, 0.911] | 0.9311 |
| S-1s | SVM-RBF | 0.9045 ± 0.003 | [0.901, 0.910] | 0.9288 |
| S-1s | SVM-polinomial | 0.9027 ± 0.002 | [0.900, 0.904] | 0.9273 |
| S-1s | MLP | 0.8776 ± 0.004 | [0.871, 0.883] | 0.9086 |
| S-1s | SVM-lineal | 0.8636 ± 0.008 | [0.854, 0.875] | 0.8981 |
| S-1s | kNN | 0.8315 ± 0.005 | [0.824, 0.839] | 0.8725 |
| G-1s | SVM-RBF | 0.6441 ± 0.002 | [0.642, 0.646] | 0.7333 |
| G-1s | SVM-lineal | 0.6286 ± 0.001 | [0.627, 0.630] | 0.7205 |
| G-1s | MLP | 0.6261 ± 0.011 | [0.609, 0.641] | 0.7186 |
| G-1s | RandomForest | 0.6073 ± 0.005 | [0.600, 0.612] | 0.7047 |
| G-1s | SVM-polinomial | 0.5877 ± 0.002 | [0.586, 0.591] | 0.6869 |
| G-1s | kNN | 0.5733 ± 0.006 | [0.565, 0.580] | 0.6692 |

### Conjunto D — FOSP+comp+EEFGabor (SelectKBest k=60)

| Condición | Clasificador | Kappa | IC95% | Acc |
|---|---|---:|---|---:|
| S-1s | SVM-RBF | 0.9078 ± 0.003 | [0.905, 0.912] | 0.9311 |
| S-1s | RandomForest | 0.9052 ± 0.004 | [0.900, 0.911] | 0.9292 |
| S-1s | SVM-polinomial | 0.9051 ± 0.005 | [0.898, 0.911] | 0.9291 |
| S-1s | kNN | 0.8827 ± 0.004 | [0.875, 0.888] | 0.9121 |
| S-1s | MLP | 0.8775 ± 0.005 | [0.874, 0.886] | 0.9084 |
| S-1s | SVM-lineal | 0.8768 ± 0.003 | [0.873, 0.881] | 0.9079 |
| G-1s | SVM-RBF | 0.6077 ± 0.002 | [0.606, 0.611] | 0.7031 |
| G-1s | SVM-polinomial | 0.5987 ± 0.001 | [0.597, 0.600] | 0.6971 |
| G-1s | SVM-lineal | 0.5966 ± 0.005 | [0.591, 0.603] | 0.6941 |
| G-1s | MLP | 0.5960 ± 0.002 | [0.593, 0.599] | 0.6939 |
| G-1s | RandomForest | 0.5952 ± 0.007 | [0.589, 0.605] | 0.6945 |
| G-1s | kNN | 0.5748 ± 0.005 | [0.566, 0.581] | 0.6749 |

### Conjunto E — FOSP+comp+EEFGabor (Fisher Vector)

| Condición | Clasificador | Kappa | IC95% | Acc |
|---|---|---:|---|---:|
| S-1s | SVM-RBF | 0.9246 ± 0.006 | [0.917, 0.932] | 0.9439 |
| S-1s | MLP | 0.9203 ± 0.003 | [0.916, 0.924] | 0.9405 |
| S-1s | SVM-polinomial | 0.9182 ± 0.005 | [0.910, 0.922] | 0.9390 |
| S-1s | kNN | 0.9174 ± 0.006 | [0.910, 0.928] | 0.9381 |
| S-1s | SVM-lineal | 0.9102 ± 0.004 | [0.905, 0.916] | 0.9331 |
| S-1s | RandomForest | 0.8953 ± 0.002 | [0.893, 0.899] | 0.9219 |
| G-1s | MLP | 0.5455 ± 0.013 | [0.528, 0.561] | 0.6550 |
| G-1s | SVM-lineal | 0.5451 ± 0.008 | [0.533, 0.554] | 0.6578 |
| G-1s | kNN | 0.5425 ± 0.032 | [0.494, 0.580] | 0.6465 |
| G-1s | RandomForest | 0.5034 ± 0.004 | [0.498, 0.510] | 0.6212 |
| G-1s | SVM-RBF | 0.4957 ± 0.010 | [0.486, 0.512] | 0.6265 |
| G-1s | SVM-polinomial | 0.4382 ± 0.036 | [0.402, 0.492] | 0.5811 |

**Nota cualitativa (multiclase, ya visible con estos números):** la
caída S-1s → G-1s es grande y consistente en los 4 conjuntos —
p. ej. conjunto B, SVM-RBF: Kappa 0.9133 → 0.6156 (Δ ≈ 0.30). Esto va
en la misma dirección que ya reportaba el paper (leakage infla las
métricas), pero con la agrupación real por grabación (M=73) en vez de
la agrupación degenerada anterior (M=2268) — los números absolutos van
a diferir de `results/leakage_stats.csv` actual (que sigue siendo el de
antes del fix). La corrida oficial del colega es la que hay que citar,
no esta.

## Binario (B, C, D completos — E no alcanzó a correr)

Datos crudos disponibles en `notebooks/colab_run.ipynb` (celda de
`leakage`, output) para B/C/D si hace falta la tabla completa —no se
transcribe aquí para no duplicar de más; lo relevante es que **falta
el conjunto E completo** en modo binario, y **falta toda la capa
estadística (Wilcoxon/Bonferroni/figuras) de las 8 combinaciones**, así
que no hay número de "significativos tras Bonferroni" que reportar de
esta corrida.

## Qué falta (lo que sí tiene que hacer la corrida oficial)

- Binario / conjunto E completo.
- Pruebas de significancia + corrección de Bonferroni familywise sobre
  las 48 comparaciones.
- Figuras (matrices de confusión, ROC).
- `results/leakage_results.csv` y `results/leakage_stats.csv` reales,
  actualizados con el fix de `_recording_id()`.
