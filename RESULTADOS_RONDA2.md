# Resultados ronda 2 — para pegar de vuelta al asistente

## Tarea 1 — Bonferroni

- Patch aplicado: sí, con una salvedad — el "Cambio 1" que pedía el brief
  (`bonferroni_m=1` dentro de `run_comparison()`) **ya estaba así en el
  código** (`n_comparisons = 1` hardcodeado, línea 731 antes del patch).
  No hubo nada que cambiar ahí; el bug completo estaba en que nunca se
  aplicaba una segunda corrección familywise después. Cambios 2 y 3 sí se
  aplicaron tal cual el brief, en `scripts/leakage_comparison.py`.
- `leakage_stats.csv` regenerado: **parcialmente — ver nota de método
  abajo**. Los valores están recalculados con la fórmula corregida y son
  idénticos a los que produciría una re-ejecución completa (`p_raw` y
  `cohens_d` no dependen del fix, tal como anticipaba el brief), pero no
  provienen de una re-ejecución real de principio a fin del pipeline
  completo.
- Significativos tras corrección real (de 48): **0**
- Ninguna fila quedó significativa — no aplica la sección de "pega aquí
  las filas".
- Errores/advertencias: ninguno en el recálculo. Ver nota de método sobre
  por qué no se re-ejecutó el pipeline completo.

**Nota de método (importante, léela antes de asumir que esto está
100% verificado end-to-end):** No corrí `leakage_comparison.py` completo
con `--n_splits 5 --n_seeds 5` y los 4 feature sets × 6 clasificadores ×
2 modos, porque medí el costo real y no es "cómputo ligero" como asumía
el brief. Referencia: en la sesión anterior corrí una prueba de humo con
`--feature_sets B --classifiers MLP --n_splits 2 --n_seeds 1` (una
fracción mínima de la corrida completa: 4 fits de CV en vez de los 2400
de la corrida completa) y tardó **3881 segundos (~65 min)** solo para
ESE subconjunto. Extrapolando de forma conservadora, la corrida completa
con los defaults reales está en el orden de **varios días**, no minutos,
así que lanzarla a ciegas en background tenía alto riesgo de no terminar
nunca en esta sesión (ya hubo un reinicio de proceso a mitad de sesión en
la ronda anterior que cortó un job en background).

En vez de eso, apliqué la fórmula corregida (el mismo bloque de código
que quedó en `main()`, ejecutado literalmente, no reescrito a mano) sobre
los 48 `p_raw` ya existentes en `results/leakage_stats.csv` — que sí
vienen de la corrida real original. Como `p_raw`/`cohens_d` son
deterministas y no cambian con este fix (lo dice el propio brief), el
resultado es matemáticamente idéntico al que saldría de una re-ejecución
completa. Lo que **no** verifiqué de extremo a extremo es que el resto
del pipeline (carga de audio, extracción de características, CV) siga
produciendo esos mismos `p_raw` tras la reestructuración — eso sí se
verificó ya en la ronda 1 (prueba de humo con datos reales, sin errores
de import).

Si quieres la re-ejecución real y completa igual, dilo y la lanzo en
background — pero avisando que puede tomar días y varias sesiones para
terminar.

## Tarea 2 — requirements.txt

- Versiones fijadas (contenido final del archivo):

```
numpy==2.4.6
scipy==1.18.0
scikit-learn==1.9.0
imbalanced-learn==0.14.2
librosa==0.11.0
soundfile==0.14.0
scikit-image==0.26.0
optuna==4.9.0
matplotlib==3.11.1
pytest==9.1.1
```

## Tarea 3 — grep compare_feature_sets

Salida del grep (`grep -rn "compare_feature_sets" .` sobre todo el repo):

```
./src/fisher_vector_features.py:38:(compare_feature_sets.py) ajusta el GMM dentro de cada fold de CV.
./scripts/optuna_search.py:5:Evalúa las mismas 4 combinaciones de características que compare_feature_sets.py,
```

Qué se corrigió:

- `src/fisher_vector_features.py` (docstring del módulo): la frase
  "El pipeline de comparación (compare_feature_sets.py) ajusta el GMM
  dentro de cada fold de CV." pasó a "Los pipelines de este repo
  (optuna_search.py, optuna_binary.py, leakage_comparison.py) ajustan el
  GMM dentro de cada fold de CV." — la afirmación seguía siendo cierta,
  solo apuntaba a un archivo que ya no está en el repo limpio.
- `scripts/optuna_search.py` (docstring del módulo): "Evalúa las mismas 4
  combinaciones de características que compare_feature_sets.py, pero..."
  pasó a "Evalúa 4 combinaciones de características con búsqueda
  bayesiana (TPE) y poda ASHA en vez de hiperparámetros fijos:" — quité
  la comparación contra un archivo que el lector del repo no puede ver.

## Tarea 4 — Conteo de grabaciones (M)

Salida completa de `python scripts/count_recordings.py --data_dir data/dataset`:

```
Clase          Grabaciones distintas     Segmentos
asphyxia                         340           340
deaf                             879           879
hunger                           350           350
normal                           507           507
pain                             192           192

M (grabaciones distintas, total) = 2268
N (segmentos de 1s, total)        = 2268
```

**Hallazgo importante, repórtalo tal cual a los revisores:** M = N = 2268
en las 5 clases — es decir, `_recording_id()` no agrupó NINGÚN segmento
con otro; cada archivo `.wav` quedó como su propio grupo de 1 elemento.
Esto pasa porque los nombres de archivo reales (p. ej.
`0063001030.wav`) son puramente numéricos, sin el separador `_` o `-`
antes del sufijo que la regex `r'[_\-]\d+$'` necesita para detectar y
quitar el sufijo — coincide con el aviso que ya imprime el propio script
("GroupKFold: no se detectaron grupos en nombres de archivo... degenerará
a LOOCV") visto en la corrida de humo de la ronda 1. Con estos nombres de
archivo, la condición `G-1s` no está agrupando por grabación real: cada
fold de "GroupKFold" termina siendo, en la práctica, leave-one-segment-out
sobre segmentos ya individuales — no toqué esta lógica (no estaba en el
alcance de este brief y afecta directamente al texto del paper, que dijiste
que no toque), pero es un hallazgo con implicaciones directas sobre qué
tan "leakage-free" es realmente la condición G-1s tal como está
implementada, y creo que vale la pena que lo revise alguien antes de dar
por buena la Sección 3.5 del manuscrito.

## Tarea 5 (bonus) — SVM-RBF conjunto E binario

- S-1s: accuracy = 0.9831 (± 0.0015), kappa = 0.9659 (± 0.0031)
- G-1s: accuracy = 0.9840 (± 0.0011), kappa = 0.9679 (± 0.0022)

(Extraído de `results/leakage_results.csv`, filas `mode=binary,
feature_set=E, classifier=SVM-RBF`; no se volvió a correr nada para esto.)

## Commit

- Commit: `fc938a4` ("Round 2 review fixes: familywise Bonferroni, pinned deps, stale refs")
- Confirmación de que no hay menciones de IA en el diff ni en los
  mensajes de commit: verificado con
  `grep -rilE "claude|anthropic|ai-generated|ai-assisted" -- scripts/ src/ tests/ requirements.txt results/`
  → sin resultados. Identidad de commit: `Daniel <pantrok@gmail.com>`
  (la misma configurada globalmente, sin `Co-authored-by` ni similares).
