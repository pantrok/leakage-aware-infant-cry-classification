# Leakage-aware Evaluation of Infant Cry Classification: A Robust Comparison of Acoustic Feature Representations

This repository contains the code accompanying the paper submitted to
*Pattern Analysis and Applications* (Springer):

> **Leakage-aware Evaluation of Infant Cry Classification: A Robust Comparison of Acoustic Feature Representations**
> Ricardo Ramos-Aguilar<sup>1,2</sup>, J. Arturo Olvera-López<sup>1</sup>, Daniel Sánchez-Ruiz<sup>2</sup>, Eric Ramos-Aguilar<sup>2</sup>
> <sup>1</sup>Benemérita Universidad Autónoma de Puebla (BUAP), Puebla, México
> <sup>2</sup>Instituto Politécnico Nacional (IPN), Tlaxcala, México

## What this is

A pipeline for classifying neonatal cry pathologies (asphyxia, deafness,
hunger, normal, pain) on the Baby Chillanto Infant Cry Database, comparing
four progressively enriched acoustic feature sets (complementary + Gabor
features, with and without FOSP, with fixed or Fisher Vector-encoded
dimensionality reduction) across six classifiers, under two evaluation
protocols:

- **Multiclass** and **binary** (healthy vs. pathology) classification with
  Optuna (TPE + ASHA) hyperparameter search.
- A **leakage-aware comparison** quantifying how much reported performance
  is inflated by segment-level data leakage: stratified K-fold over 1-second
  fragments (`S-1s`, the common practice in prior work, which lets fragments
  from the same recording land in both train and test) vs. group K-fold by
  recording (`G-1s`, which keeps every fragment of a recording in a single
  fold), with bootstrap confidence intervals, paired Wilcoxon signed-rank
  tests (Bonferroni-corrected), and Cohen's d effect sizes.

## Repository layout

```
├── src/            shared feature-extraction / preprocessing / classifier modules
├── scripts/        the three entry points (also runnable standalone) + count_recordings.py
├── main.py         single dispatcher: python main.py {multiclass,binary,leakage}
├── tests/          unit tests for the leakage-comparison script
├── results/        CSV/PNG outputs cited in the paper (copied as-is, not regenerated)
├── data/           README documenting the expected dataset layout (no data)
└── notebooks/      colab_run.ipynb — recommended way to reproduce without a local GPU
```

## Dataset

This pipeline uses the **Baby Chillanto Infant Cry Database**, collected and
maintained by the Instituto Nacional de Astrofísica, Óptica y Electrónica
(INAOE), Mexico. **The dataset is not distributed in this repository** — it
must be requested directly from the database's custodians / original
authors. See [`data/README.md`](data/README.md) for the exact folder
structure the code expects once you have it.

Original references for the database:

- Reyes-Galaviz OF, Cano-Ortiz SD, Reyes-García CA (2008) Evolutionary-neural
  system to classify infant cry units for pathologies identification in
  recently born babies. In: Seventh Mexican International Conference on
  Artificial Intelligence (MICAI). IEEE, pp 330–335.
- Rosales-Pérez A, Reyes-García CA, Gonzalez JA, Reyes-Galaviz OF, Escalante
  HJ, Orlandi S (2015) Classifying infant cry patterns by the genetic
  selection of a fuzzy model. Biomed Signal Process Control 17:38–46.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python ≥ 3.9. GPU acceleration is optional:

```bash
pip install --extra-index-url=https://pypi.nvidia.com cuml-cu12
```

speeds up SVM/kNN/RandomForest via cuML (Linux/WSL2 only, no native Windows
build); PyTorch+CUDA accelerates the MLP automatically if available, no
extra install needed. The scikit-learn/CPU fallback in
[`src/gpu_classifiers.py`](src/gpu_classifiers.py) is what reproduces the
exact numbers reported in the paper; see that module's docstring for details.

## Usage

All three experiments share the same entry point:

```bash
python main.py multiclass --data_dir data/dataset [flags]
python main.py binary     --data_dir data/dataset [flags]
python main.py leakage    --data_dir data/dataset [flags]
```

`main.py` forwards every flag after the subcommand straight to the
corresponding script's own argument parser (run `python main.py multiclass
--help` to see them all). Each script also runs standalone:

```bash
python scripts/optuna_search.py --data_dir data/dataset ...
python scripts/optuna_binary.py --data_dir data/dataset ...
python scripts/leakage_comparison.py --data_dir data/dataset ...
```

No local GPU? See [`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb)
for the recommended way to run this from Google Colab with the dataset on
Google Drive.

Before a full `leakage` run, `python scripts/count_recordings.py
--data_dir data/dataset` is a cheap sanity check that the recording-level
grouping used by the `G-1s` condition is working (should print `M = 73`).

## License

Code is released under the [MIT License](LICENSE). This license covers the
source code only — it does **not** cover the Baby Chillanto dataset (see
[Dataset](#dataset) above).
