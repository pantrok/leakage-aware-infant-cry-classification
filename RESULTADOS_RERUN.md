# Resultados de la re-ejecución con recording_id corregido

## Actualización — decisión de GPU tomada, código listo

Daniel eligió la opción 2 (wiring GPU real) para el bloqueo que dejé
abajo. Ya está hecho y pusheado, commit `aab912e`:
`scripts/leakage_comparison.py` ahora construye sus 6 clasificadores vía
`src/gpu_classifiers.py` (cuML/RAPIDS para SVM/kNN/RandomForest,
PyTorch+CUDA para el MLP), con el mismo fallback automático a
sklearn/CPU que ya usan `optuna_search.py`/`optuna_binary.py` cuando no
hay GPU. Verificado en esta máquina (sin GPU/cuML) que el fallback CPU
sigue devolviendo exactamente las mismas clases de sklearn que antes, y
que `pytest tests/` sigue en 2/2.

`notebooks/colab_run.ipynb` ya incluye: instrucción de activar runtime
GPU, celda opcional de `%pip install cuml-cu12`, y la nota de que cuML
puede dar números ligeramente distintos a sklearn/CPU (no es un bug).

**Sigue sin correrse la corrida completa** — eso solo se puede verificar
de verdad en un runtime con GPU real, que esta máquina no tiene. Queda
lista para que se lance desde el notebook en Colab (celda de `leakage`,
sección 6). El resto de este documento (tareas 1-3 originales) se dejó
tal cual quedó cuando se detectó el bloqueo, como registro.

## Tarea 1 — Fix aplicado

Sí, tal cual el brief, en ambos archivos:

- `scripts/leakage_comparison.py`: `_recording_id()` ahora reconoce el
  esquema de 10 dígitos `PPPPSSSCCC` (regex `^(\d{4})\d{6}$`, regresa
  `PPPP`), con fallback a la heurística anterior de sufijo con
  separador para nombres que no matcheen.
- `scripts/count_recordings.py`: mismo cambio, para que quede
  consistente con `leakage_comparison.py`.
- Extra verificado antes de dar por buena la fórmula: los 73 prefijos
  de 4 dígitos son **globalmente únicos entre las 5 clases** (cero
  colisiones cruzadas) — confirmado con un conteo directo sobre los
  nombres de archivo reales, así que usar solo `PPPP` sin distinguir
  clase no mezcla grabaciones de clases distintas por accidente.
- `pytest tests/` sigue pasando (2/2) tras el cambio.

Commit: `fa5e845` ("Fix _recording_id() to match the real Baby Chillanto
filename scheme").

## Tarea 2 — Verificación de M

Salida completa de
`python scripts/count_recordings.py --data_dir "../Bebes Grid indiv - Optuna - 2/Data"`
(la ruta real, fuera del repo):

```
Clase          Grabaciones distintas     Segmentos
asphyxia                           6           340
deaf                                6           879
hunger                             33           350
normal                              5           507
pain                                23           192

M (grabaciones distintas, total) = 73
N (segmentos de 1s, total)        = 2268
```

**M = 73, coincide exactamente** con lo esperado del diagnóstico
(6/6/33/5/23). Se continuó a lo que se pudo de la Tarea 3.

## Tarea 3 — Estado de la corrida completa

**No se corrió aquí — ni siquiera se lanzó.** Antes de eso hay un
hallazgo que cambia el plan y que necesito que confirmes:

### Hallazgo: `leakage_comparison.py` no usa GPU hoy, con o sin Colab

Revisé los imports de `scripts/leakage_comparison.py`: construye sus
clasificadores directo desde `sklearn` (`SVC`, `KNeighborsClassifier`,
`MLPClassifier`, `RandomForestClassifier`), **sin pasar por
`src/gpu_classifiers.py`** (que es el módulo con el fallback
cuML/PyTorch+CUDA → sklearn/CPU). Ese módulo solo lo usan
`optuna_search.py` y `optuna_binary.py`. Es decir: correr
`leakage_comparison.py` en un runtime GPU de Colab, tal como está el
código hoy, tardaría **exactamente lo mismo** que en CPU — cero
aceleración. El mensaje que pediste ("hay que ocupar recursos de GPU")
no se cumple todavía con el código actual.

Esto explica también por qué la estimación de "varios días" de la
ronda anterior no cambia solo por mudarse a Colab: esa estimación ya
asumía CPU, y ya reflejaba correctamente lo que este script hace hoy.

**Dos caminos, necesito que elijas uno antes de seguir:**

1. **Correrlo tal cual, en Colab, en CPU** (dejar `leakage_comparison.py`
   sin tocar). Es simple y no toca ninguna lógica más allá del fix de
   `_recording_id()` ya aplicado, pero el tiempo estimado sigue siendo
   del orden de días — Colab gratuito además desconecta sesiones
   inactivas (~90 min sin interacción, límite duro ~12h), así que una
   corrida de días necesitaría Colab Pro/Pro+ con ejecución en segundo
   plano, o partirla en corridas más chicas por `--feature_sets`/`--modes`
   que quepan en una sesión.
2. **Modificar `leakage_comparison.py` para que construya sus
   clasificadores vía `src/gpu_classifiers.py`** (mismo patrón que ya
   usan `optuna_search.py`/`optuna_binary.py`), para que sí aproveche
   GPU en Colab. Esto SÍ es un cambio de lógica más allá del fix de
   `_recording_id()` que me pediste explícitamente en este brief, y
   tiene una consecuencia a tener en cuenta: `gpu_classifiers.py` avisa
   en su propio docstring que **cuML no es binariamente idéntico a
   sklearn** (solvers/inicializaciones internas distintos) — los
   números de la condición GPU podrían diferir ligeramente de los que
   ya están citados en el paper (que vienen de sklearn/CPU). No lo hice
   sin confirmar contigo dado que rondas anteriores fueron explícitas
   en "no toques la lógica de leakage_comparison.py".

No voy a lanzar la corrida cara hasta que me digas cuál de las dos
quieres (o si prefieres algo distinto, como reducir el grid con
`--feature_sets`/`--classifiers`/`--modes` para que quepa en una sesión
de Colab gratuito).

### Comando listo para cuando se decida

```bash
python scripts/leakage_comparison.py \
    --data_dir "/content/drive/MyDrive/Publicaciones/Chillanto CIARP/Bebes Grid indiv - Optuna - 2/Data" \
    --n_splits 5 --n_seeds 5 --balanced \
    --csv_results results/leakage_results.csv \
    --csv_stats results/leakage_stats.csv \
    --plot_dir results/leakage_plots
```

(o la celda equivalente ya en `notebooks/colab_run.ipynb`, sección "6.
Run the three experiments" → celda de `leakage`). Notas para quien lo
corra (tú o tu colega):

- Dependencias ya fijadas en `requirements.txt` (`%pip install -r
  requirements.txt` en el notebook).
- La ruta de datos real ya está fuera del repo, en Drive, en
  `Bebes Grid indiv - Optuna - 2/Data/` (sibling folder al repo) — el
  notebook ya apunta ahí por default, no hace falta copiar nada a
  `data/dataset/`.
- Antes de lanzar la corrida cara, correr
  `python scripts/count_recordings.py --data_dir "<esa misma ruta>"` y
  confirmar que da `M = 73` (ya viene como celda 5 del notebook
  actualizado).
- Cuando termine (aquí o en la máquina del colega), el fix de
  Bonferroni de la ronda anterior ya está integrado en `main()` — no
  hace falta tocar nada adicional para eso.

Nada de esto se ejecutó todavía, así que no hay tabla de Kappa/Accuracy
nueva que reportar, ni conteo de significativos nuevo, ni forma de saber
si el aviso de "GroupKFold degenerará a LOOCV" desaparece — todo eso
queda pendiente de la corrida real.

## Commit

- `fa5e845` — fix de `_recording_id()` en ambos scripts + actualización
  del notebook de Colab (rutas reales de Drive, celda de sanity-check de
  M=73, nota de GPU).
- Confirmado sin menciones de IA:
  `grep -rilE "claude|anthropic|ai-generated|ai-assisted" scripts/leakage_comparison.py scripts/count_recordings.py notebooks/colab_run.ipynb`
  → sin resultados. Identidad: `Daniel <pantrok@gmail.com>`.
