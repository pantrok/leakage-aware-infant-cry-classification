# Resultados de la re-ejecución de leakage (máquina con GPU real)

## Entorno
- SO / GPU usada: Windows 11, NVIDIA GeForce RTX 5060 Ti (nativo, no WSL2)
- `cuml-cu12` instalado: no — cuML no tiene build nativo de Windows (solo Linux/WSL2), tal como advierte el propio script. SVM/kNN/RandomForest corrieron por el fallback scikit-learn/CPU; el MLP sí usó GPU vía PyTorch+CUDA (`torch==2.11.0+cu128`, confirmado con `torch.cuda.is_available() == True`, dispositivo "NVIDIA GeForce RTX 5060 Ti").
- Salida exacta de `report_backend()` al inicio de la corrida:

```
[INFO] Backend de cómputo: scikit-learn (CPU). cuML no disponible -- instala con 'pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12' si tienes GPU NVIDIA+CUDA y quieres acelerar SVM/kNN (nota: cuML no tiene build para Windows nativo, solo Linux/WSL2).
[INFO] PyTorch detecta GPU CUDA disponible para el MLP.
```

## Verificación M=73

Salida completa de `count_recordings.py`:

```
Clase          Grabaciones distintas     Segmentos
asphyxia                           6           340
deaf                                6           879
hunger                             33           350
normal                              5           507
pain                               23           192

M (grabaciones distintas, total) = 73
N (segmentos de 1s, total)        = 2268
```

¿Coincidió exactamente? **Sí**, coincide con la tabla esperada del brief, tanto en el desglose por clase como en M=73 y N=2268.

## Corrida de leakage

- ¿Se cortó en algún momento? **Sí, pero no por causas del pipeline en sí**: el primer intento crasheó de inmediato (antes de procesar nada) por un `UnicodeEncodeError` de la consola Windows (cp1252) al imprimir el carácter `─` (U+2500) del encabezado de cada conjunto de características — un problema puramente de encoding de terminal en Windows, no de la lógica del script. Se resolvió corriendo con `PYTHONUTF8=1 PYTHONIOENCODING=utf-8` (sin tocar código). Un segundo arranque se detuvo manualmente a los ~30s porque el stdout estaba bufferizado y no permitía monitorear progreso en vivo (se reinició con `-u`/`PYTHONUNBUFFERED=1`); en ese punto no se había procesado ningún dato todavía, así que no hubo checkpoint que perder. **La corrida real y completa (la que produjo todos los resultados de abajo) corrió de principio a fin sin cortes ni necesidad de resume** — no aparece ningún `[RESUME]` ni `[SKIP` en el log porque nunca hizo falta.
- Tiempo total y por combinación (8 combinaciones: 2 modos × 4 conjuntos):

| Modo | Conjunto | Tiempo |
|---|---|---:|
| multiclass | B | 717.2 s (~12.0 min) |
| multiclass | C | 806.5 s (~13.4 min) |
| multiclass | D | 658.7 s (~11.0 min) |
| multiclass | E | 4616.2 s (~77.0 min) |
| binary | B | 505.2 s (~8.4 min) |
| binary | C | 616.8 s (~10.3 min) |
| binary | D | 586.0 s (~9.8 min) |
| binary | E | 4008.8 s (~66.8 min) |

Tiempo total de cómputo del grid: **12,515.4 s (~3 h 28 min)**, más el tiempo de instalación de dependencias (aparte). Notablemente más rápido que la referencia de Colab (~17-21 min para B/C/D, ~3 h por E) — probablemente por ser GPU dedicada sin las limitaciones de una sesión compartida de Colab free, y por diferencias de hardware/contención de CPU en la extracción de características.

- Cualquier error o warning durante la corrida: **ninguno**, aparte del `UnicodeEncodeError` inicial ya descrito (resuelto antes de la corrida "oficial"). Se buscó explícitamente `error`, `warning`, `Traceback` en el log completo y no aparece nada más.
- ¿Apareció el aviso "GroupKFold degenerará a LOOCV" para alguna combinación? **No**, no aparece en ningún punto del log — confirmado con búsqueda explícita.

## Resultados

Tabla completa Kappa/Accuracy (media ± std, IC95%) por (modo, conjunto, condición, clasificador), tal cual la imprime el script en consola:

### MULTICLASE (5 clases)

**[B] comp+EEFGabor (sin FOSP) — S-1s: StratifiedKFold aleatorio (baseline literatura)**
```
Clasificador       Kappa    ±         IC95%     Acc  Acc_std
------------------------------------------------------------
RandomForest      0.9144  0.003 [0.909,0.919]  0.9362   0.0025
SVM-RBF           0.9106  0.002 [0.908,0.914]  0.9333   0.0018
SVM-polinomial    0.9045  0.003 [0.902,0.909]  0.9287   0.0020
MLP               0.8802  0.004 [0.874,0.886]  0.9106   0.0033
SVM-lineal        0.8719  0.004 [0.864,0.875]  0.9044   0.0033
kNN               0.8452  0.004 [0.841,0.852]  0.8834   0.0030
```

**[B] comp+EEFGabor (sin FOSP) — G-1s: GroupKFold por grabación (leakage controlado)**
```
Clasificador       Kappa    ±         IC95%     Acc  Acc_std
------------------------------------------------------------
SVM-lineal        0.6528  0.002 [0.650,0.657]  0.7392   0.0019
MLP               0.6273  0.005 [0.620,0.634]  0.7189   0.0036
SVM-RBF           0.6172  0.002 [0.615,0.620]  0.7115   0.0013
RandomForest      0.5910  0.006 [0.582,0.600]  0.6922   0.0048
SVM-polinomial    0.5829  0.002 [0.580,0.586]  0.6832   0.0017
kNN               0.5610  0.008 [0.547,0.568]  0.6622   0.0066
```

**[C] FOSP+comp+EEFGabor (sin reduc.) — S-1s**
```
Clasificador       Kappa    ±         IC95%     Acc  Acc_std
------------------------------------------------------------
RandomForest      0.9090  0.003 [0.904,0.913]  0.9321   0.0022
SVM-RBF           0.9049  0.002 [0.902,0.907]  0.9291   0.0013
SVM-polinomial    0.9000  0.005 [0.895,0.908]  0.9253   0.0034
MLP               0.8715  0.007 [0.864,0.883]  0.9041   0.0055
SVM-lineal        0.8630  0.004 [0.859,0.868]  0.8977   0.0027
kNN               0.8298  0.010 [0.814,0.841]  0.8712   0.0076
```

**[C] FOSP+comp+EEFGabor (sin reduc.) — G-1s**
```
Clasificador       Kappa    ±         IC95%     Acc  Acc_std
------------------------------------------------------------
SVM-RBF           0.6439  0.002 [0.642,0.646]  0.7334   0.0012
SVM-lineal        0.6376  0.001 [0.637,0.639]  0.7280   0.0006
MLP               0.6251  0.007 [0.614,0.633]  0.7178   0.0054
RandomForest      0.6068  0.003 [0.603,0.610]  0.7044   0.0026
SVM-polinomial    0.5880  0.002 [0.585,0.591]  0.6873   0.0018
kNN               0.5705  0.008 [0.558,0.579]  0.6671   0.0062
```

**[D] FOSP+comp+EEFGabor (k=60) — S-1s**
```
Clasificador       Kappa    ±         IC95%     Acc  Acc_std
------------------------------------------------------------
RandomForest      0.9067  0.003 [0.904,0.911]  0.9303   0.0021
SVM-RBF           0.9051  0.001 [0.903,0.906]  0.9291   0.0008
SVM-polinomial    0.9027  0.003 [0.898,0.907]  0.9272   0.0023
kNN               0.8854  0.008 [0.874,0.894]  0.9141   0.0059
MLP               0.8783  0.005 [0.871,0.883]  0.9090   0.0035
SVM-lineal        0.8749  0.003 [0.870,0.878]  0.9065   0.0024
```

**[D] FOSP+comp+EEFGabor (k=60) — G-1s**
```
Clasificador       Kappa    ±         IC95%     Acc  Acc_std
------------------------------------------------------------
SVM-lineal        0.6195  0.004 [0.616,0.625]  0.7129   0.0030
SVM-RBF           0.6104  0.004 [0.607,0.616]  0.7054   0.0028
SVM-polinomial    0.6047  0.002 [0.602,0.607]  0.7018   0.0014
MLP               0.5989  0.005 [0.593,0.606]  0.6961   0.0037
RandomForest      0.5974  0.005 [0.591,0.603]  0.6961   0.0036
kNN               0.5721  0.003 [0.567,0.575]  0.6724   0.0025
```

**[E] FOSP+comp+EEFGabor (FisherVec) — S-1s**
```
Clasificador       Kappa    ±         IC95%     Acc  Acc_std
------------------------------------------------------------
SVM-RBF           0.9166  0.006 [0.909,0.926]  0.9380   0.0045
MLP               0.9164  0.003 [0.913,0.920]  0.9376   0.0023
kNN               0.9111  0.008 [0.897,0.920]  0.9333   0.0064
SVM-polinomial    0.9076  0.013 [0.886,0.919]  0.9310   0.0099
SVM-lineal        0.9055  0.004 [0.902,0.912]  0.9295   0.0031
RandomForest      0.9009  0.005 [0.893,0.907]  0.9261   0.0039
```

**[E] FOSP+comp+EEFGabor (FisherVec) — G-1s**
```
Clasificador       Kappa    ±         IC95%     Acc  Acc_std
------------------------------------------------------------
MLP               0.5492  0.017 [0.524,0.570]  0.6563   0.0134
SVM-lineal        0.5451  0.007 [0.534,0.552]  0.6580   0.0061
kNN               0.5369  0.027 [0.492,0.561]  0.6417   0.0224
RandomForest      0.5155  0.012 [0.503,0.535]  0.6311   0.0090
SVM-RBF           0.5112  0.003 [0.506,0.514]  0.6380   0.0032
SVM-polinomial    0.4455  0.011 [0.431,0.460]  0.5873   0.0090
```

### BINARIO (healthy vs pathology)

healthy ← [hunger, normal, pain] · pathology ← [asphyxia, deaf]

**[B] comp+EEFGabor (sin FOSP) — S-1s**
```
Clasificador       Kappa    ±         IC95%     Acc    Sens    Spec     AUC
---------------------------------------------------------------------------
SVM-RBF           0.9548  0.005 [0.948,0.962]  0.9776  0.9934  0.9592  0.9966
RandomForest      0.9506  0.004 [0.944,0.956]  0.9755  0.9916  0.9567  0.9976
SVM-polinomial    0.9499  0.003 [0.948,0.953]  0.9751  0.9829  0.9661  0.9960
MLP               0.9441  0.005 [0.939,0.952]  0.9722  0.9800  0.9632  0.9964
kNN               0.9303  0.003 [0.927,0.935]  0.9653  0.9652  0.9655  0.9938
SVM-lineal        0.8875  0.003 [0.883,0.891]  0.9441  0.9483  0.9392  0.9874
```

**[B] comp+EEFGabor (sin FOSP) — G-1s**
```
Clasificador       Kappa    ±         IC95%     Acc    Sens    Spec     AUC
---------------------------------------------------------------------------
kNN               0.7233  0.002 [0.722,0.726]  0.8614  0.8231  0.9058  0.9027
SVM-RBF           0.7118  0.001 [0.710,0.714]  0.8560  0.8354  0.8799  0.9160
RandomForest      0.7108  0.008 [0.703,0.722]  0.8555  0.8335  0.8810  0.9166
SVM-polinomial    0.7081  0.002 [0.704,0.711]  0.8535  0.8077  0.9068  0.8964
MLP               0.7039  0.007 [0.692,0.713]  0.8519  0.8228  0.8856  0.9244
SVM-lineal        0.5480  0.005 [0.543,0.556]  0.7750  0.7824  0.7663  0.8435
```

**[C] FOSP+comp+EEFGabor (sin reduc.) — S-1s**
```
Clasificador       Kappa    ±         IC95%     Acc    Sens    Spec     AUC
---------------------------------------------------------------------------
SVM-polinomial    0.9482  0.003 [0.944,0.952]  0.9743  0.9759  0.9724  0.9954
RandomForest      0.9477  0.003 [0.945,0.952]  0.9741  0.9910  0.9544  0.9971
SVM-RBF           0.9466  0.004 [0.939,0.951]  0.9735  0.9921  0.9520  0.9959
MLP               0.9338  0.006 [0.927,0.943]  0.9671  0.9775  0.9550  0.9958
kNN               0.9200  0.004 [0.914,0.926]  0.9601  0.9472  0.9752  0.9919
SVM-lineal        0.8790  0.009 [0.867,0.891]  0.9399  0.9437  0.9354  0.9872
```

**[C] FOSP+comp+EEFGabor (sin reduc.) — G-1s**
```
Clasificador       Kappa    ±         IC95%     Acc    Sens    Spec     AUC
---------------------------------------------------------------------------
kNN               0.7412  0.006 [0.734,0.751]  0.8701  0.8197  0.9287  0.9237
SVM-RBF           0.7189  0.002 [0.716,0.722]  0.8597  0.8448  0.8770  0.9288
RandomForest      0.7142  0.005 [0.706,0.719]  0.8571  0.8348  0.8831  0.9272
MLP               0.7108  0.004 [0.705,0.717]  0.8553  0.8249  0.8906  0.9391
SVM-polinomial    0.6919  0.004 [0.685,0.696]  0.8450  0.7821  0.9180  0.9116
SVM-lineal        0.5928  0.000 [0.593,0.593]  0.7972  0.8007  0.7931  0.8693
```

**[D] FOSP+comp+EEFGabor (k=60) — S-1s**
```
Clasificador       Kappa    ±         IC95%     Acc    Sens    Spec     AUC
---------------------------------------------------------------------------
SVM-RBF           0.9367  0.003 [0.933,0.939]  0.9686  0.9885  0.9455  0.9934
RandomForest      0.9346  0.002 [0.932,0.936]  0.9675  0.9811  0.9518  0.9959
kNN               0.9294  0.006 [0.920,0.937]  0.9649  0.9644  0.9655  0.9928
SVM-polinomial    0.9290  0.002 [0.926,0.932]  0.9648  0.9906  0.9348  0.9928
MLP               0.9276  0.003 [0.924,0.932]  0.9640  0.9731  0.9535  0.9949
SVM-lineal        0.8821  0.006 [0.874,0.888]  0.9414  0.9532  0.9277  0.9831
```

**[D] FOSP+comp+EEFGabor (k=60) — G-1s**
```
Clasificador       Kappa    ±         IC95%     Acc    Sens    Spec     AUC
---------------------------------------------------------------------------
kNN               0.7397  0.002 [0.737,0.742]  0.8695  0.8256  0.9205  0.9044
SVM-RBF           0.7382  0.003 [0.733,0.741]  0.8691  0.8463  0.8957  0.9064
SVM-polinomial    0.7185  0.002 [0.716,0.721]  0.8593  0.8348  0.8877  0.9139
RandomForest      0.7089  0.004 [0.703,0.713]  0.8545  0.8310  0.8818  0.9181
MLP               0.6972  0.003 [0.692,0.702]  0.8483  0.8131  0.8892  0.9211
SVM-lineal        0.6464  0.004 [0.641,0.653]  0.8225  0.7747  0.8780  0.8785
```

**[E] FOSP+comp+EEFGabor (FisherVec) — S-1s**
```
Clasificador       Kappa    ±         IC95%     Acc    Sens    Spec     AUC
---------------------------------------------------------------------------
MLP               0.9700  0.003 [0.966,0.973]  0.9851  0.9879  0.9819  0.9970
SVM-RBF           0.9675  0.002 [0.965,0.971]  0.9839  0.9933  0.9729  0.9972
RandomForest      0.9522  0.005 [0.946,0.960]  0.9763  0.9926  0.9573  0.9969
SVM-lineal        0.9516  0.004 [0.945,0.955]  0.9759  0.9788  0.9725  0.9926
kNN               0.9462  0.007 [0.937,0.955]  0.9732  0.9647  0.9830  0.9950
SVM-polinomial    0.9452  0.019 [0.914,0.968]  0.9727  0.9624  0.9846  0.9959
```

**[E] FOSP+comp+EEFGabor (FisherVec) — G-1s**
```
Clasificador       Kappa    ±         IC95%     Acc    Sens    Spec     AUC
---------------------------------------------------------------------------
RandomForest      0.7418  0.007 [0.732,0.751]  0.8709  0.8440  0.9022  0.9025
MLP               0.7159  0.013 [0.704,0.737]  0.8576  0.8164  0.9054  0.9268
kNN               0.7106  0.037 [0.660,0.750]  0.8544  0.7929  0.9258  0.8997
SVM-RBF           0.6990  0.016 [0.673,0.714]  0.8499  0.8422  0.8589  0.8789
SVM-lineal        0.6853  0.015 [0.662,0.705]  0.8421  0.7956  0.8961  0.8594
SVM-polinomial    0.5793  0.021 [0.549,0.608]  0.7884  0.7306  0.8555  0.9038
```

### Significancia tras Bonferroni real (48 comparaciones, familywise, α=0.05)

**Las 48 de 48 comparaciones S-1s vs G-1s quedaron significativas** tras la corrección de Bonferroni familywise (m=48, no el no-op m=1 de antes). Tabla completa:

```
Modo        Set  Clasificador    Cond_A  Cond_B      p_raw   p_bonf  Cohen_d   Sig
----------------------------------------------------------------------------------
multiclass  B    MLP             S-1s    G-1s       0.0000   0.0000   2.1551     ✓
multiclass  B    SVM-lineal      S-1s    G-1s       0.0000   0.0000   1.8411     ✓
multiclass  B    SVM-RBF         S-1s    G-1s       0.0000   0.0000   2.0048     ✓
multiclass  B    SVM-polinomial  S-1s    G-1s       0.0000   0.0000   2.1987     ✓
multiclass  B    kNN             S-1s    G-1s       0.0000   0.0000   2.5106     ✓
multiclass  B    RandomForest    S-1s    G-1s       0.0000   0.0000   2.0715     ✓
multiclass  C    MLP             S-1s    G-1s       0.0000   0.0000   2.0498     ✓
multiclass  C    SVM-lineal      S-1s    G-1s       0.0000   0.0000   1.7548     ✓
multiclass  C    SVM-RBF         S-1s    G-1s       0.0000   0.0000   1.8815     ✓
multiclass  C    SVM-polinomial  S-1s    G-1s       0.0000   0.0000   2.1941     ✓
multiclass  C    kNN             S-1s    G-1s       0.0000   0.0000   2.4274     ✓
multiclass  C    RandomForest    S-1s    G-1s       0.0000   0.0000   2.0978     ✓
multiclass  D    MLP             S-1s    G-1s       0.0000   0.0000   1.9579     ✓
multiclass  D    SVM-lineal      S-1s    G-1s       0.0000   0.0000   1.9568     ✓
multiclass  D    SVM-RBF         S-1s    G-1s       0.0000   0.0000   2.0556     ✓
multiclass  D    SVM-polinomial  S-1s    G-1s       0.0000   0.0000   2.1962     ✓
multiclass  D    kNN             S-1s    G-1s       0.0000   0.0000   2.3764     ✓
multiclass  D    RandomForest    S-1s    G-1s       0.0000   0.0000   2.0864     ✓
multiclass  E    MLP             S-1s    G-1s       0.0000   0.0000   2.3699     ✓
multiclass  E    SVM-lineal      S-1s    G-1s       0.0000   0.0000   2.5549     ✓
multiclass  E    SVM-RBF         S-1s    G-1s       0.0000   0.0000   2.2917     ✓
multiclass  E    SVM-polinomial  S-1s    G-1s       0.0000   0.0000   3.1729     ✓
multiclass  E    kNN             S-1s    G-1s       0.0000   0.0000   2.4774     ✓
multiclass  E    RandomForest    S-1s    G-1s       0.0000   0.0000   1.9336     ✓
binary      B    MLP             S-1s    G-1s       0.0000   0.0000   1.6146     ✓
binary      B    SVM-lineal      S-1s    G-1s       0.0000   0.0013   1.5918     ✓
binary      B    SVM-RBF         S-1s    G-1s       0.0001   0.0048   1.5583     ✓
binary      B    SVM-polinomial  S-1s    G-1s       0.0000   0.0000   1.6744     ✓
binary      B    kNN             S-1s    G-1s       0.0000   0.0007   1.6456     ✓
binary      B    RandomForest    S-1s    G-1s       0.0000   0.0011   1.4724     ✓
binary      C    MLP             S-1s    G-1s       0.0000   0.0001   1.5830     ✓
binary      C    SVM-lineal      S-1s    G-1s       0.0000   0.0004   1.5087     ✓
binary      C    SVM-RBF         S-1s    G-1s       0.0005   0.0262   1.4698     ✓
binary      C    SVM-polinomial  S-1s    G-1s       0.0000   0.0000   1.7738     ✓
binary      C    kNN             S-1s    G-1s       0.0000   0.0000   1.5555     ✓
binary      C    RandomForest    S-1s    G-1s       0.0000   0.0011   1.4785     ✓
binary      D    MLP             S-1s    G-1s       0.0000   0.0000   1.5981     ✓
binary      D    SVM-lineal      S-1s    G-1s       0.0002   0.0078   1.5363     ✓
binary      D    SVM-RBF         S-1s    G-1s       0.0007   0.0352   1.4026     ✓
binary      D    SVM-polinomial  S-1s    G-1s       0.0001   0.0028   1.4882     ✓
binary      D    kNN             S-1s    G-1s       0.0000   0.0000   1.5450     ✓
binary      D    RandomForest    S-1s    G-1s       0.0000   0.0000   1.5373     ✓
binary      E    MLP             S-1s    G-1s       0.0000   0.0000   1.6732     ✓
binary      E    SVM-lineal      S-1s    G-1s       0.0000   0.0000   1.8329     ✓
binary      E    SVM-RBF         S-1s    G-1s       0.0000   0.0002   1.5145     ✓
binary      E    SVM-polinomial  S-1s    G-1s       0.0000   0.0000   2.0695     ✓
binary      E    kNN             S-1s    G-1s       0.0000   0.0000   1.7725     ✓
binary      E    RandomForest    S-1s    G-1s       0.0002   0.0115   1.4713     ✓
```

Todas las filas con `✓` (48/48). Los tamaños de efecto (Cohen's d) van de ~1.40 a ~3.17 — grandes en todos los casos, consistentes con una caída de rendimiento sistemática (no ruido) al pasar de S-1s a G-1s.

Resumen de inflación por leakage (mejor clasificador por bloque):

```
Modo        Set  Clasificador        S-1s    G-1s  Δ Kappa
------------------------------------------------------------
multiclass  B    RandomForest      0.9144  0.5910   0.3234
multiclass  C    RandomForest      0.9090  0.6068   0.3022
multiclass  D    RandomForest      0.9067  0.5974   0.3093
multiclass  E    SVM-RBF           0.9166  0.5112   0.4054
binary      B    SVM-RBF           0.9548  0.7118   0.2430
binary      C    SVM-polinomial    0.9482  0.6919   0.2563
binary      D    SVM-RBF           0.9367  0.7382   0.1985
binary      E    MLP               0.9700  0.7159   0.2541
```

- `results/leakage_results.csv`: generado (97 líneas incl. cabecera).
- `results/leakage_stats.csv`: generado (49 líneas incl. cabecera).
- `results/leakage_plots/`: **8 PNG generados** — 4 matrices de confusión (multiclase, una por conjunto B/C/D/E, mostrando el mejor clasificador de cada uno) + 4 curvas ROC (binario, una por conjunto B/C/D/E). Nota: el repo clonado ya traía 2 PNG viejos de un commit anterior (`cm_multiclass_B_MLP.png`, `cm_multiclass_E_MLP.png`, de cuando MLP era el mejor clasificador en esos bloques en una corrida previa distinta) que quedaron obsoletos porque en esta corrida el mejor clasificador de esos bloques cambió (RandomForest en B, SVM-RBF en E) — se borraron para no dejar plots que no corresponden a los números de esta corrida oficial.

## Comparación contra el parcial de Colab

Comparación multiclase (Colab sí llegó a completar las 4 combinaciones):

| Set | Clasificador | Condición | Colab (parcial) | Esta corrida | Diferencia |
|---|---|---|---:|---:|---:|
| B | SVM-RBF | S-1s | 0.9133 | 0.9106 | -0.0027 |
| B | SVM-lineal | G-1s | 0.6462 | 0.6528 | +0.0066 |
| C | RandomForest | S-1s | 0.9077 | 0.9090 | +0.0013 |
| C | SVM-RBF | G-1s | 0.6441 | 0.6439 | -0.0002 |
| D | SVM-RBF | S-1s | 0.9078 | 0.9051 | -0.0027 |
| D | SVM-lineal | G-1s | 0.6077 | 0.6195 | +0.0118 |
| E | SVM-RBF | S-1s | 0.9246 | 0.9166 | -0.0080 |
| E | MLP | G-1s | 0.5455 | 0.5492 | +0.0037 |

**Consistentes**, sin discrepancias raras: las diferencias son de ~0.001-0.012 en Kappa, del orden esperable por aleatoriedad de folds/semillas y por correr en CPU (sklearn) en vez de cuML/GPU para SVM/kNN/RandomForest (solo el MLP compartió motor GPU con Colab). La caída S-1s → G-1s tiene la misma magnitud y dirección en ambas corridas (p. ej. conjunto E: Colab Δ≈0.379, esta corrida Δ≈0.367), y el ranking de clasificadores por bloque es prácticamente el mismo. No hay ninguna discrepancia que sugiera un problema de reproducibilidad o un bug distinto al ya conocido y corregido.

No hay binario/E de Colab con qué comparar (se cortó ahí), así que esa combinación solo tiene el número de esta corrida.

## Commit

- Hash del commit: ver mensaje de confirmación tras el commit (tarea pendiente de completar en este mismo turno — el archivo se generó antes de commitear).
- Confirmación de que no hay menciones de IA en el diff ni en los mensajes de commit: pendiente de verificar en el momento del commit.
- Rama: pendiente de confirmar con el usuario antes de hacer push (ver `Qué NO hacer` del brief — no se hace push directo a `main` sin permiso explícito).
