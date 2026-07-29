"""
Estudio de leakage controlado por grabación — Baby Chillanto
=============================================================
TODO el experimento usa EXCLUSIVAMENTE las carpetas 1s_X del dataset.
Compara DOS condiciones de división sobre los mismos datos para cuantificar
la inflación de métricas por fuga de datos (data leakage) en la literatura:

  S-1s : StratifiedKFold sobre segmentos de 1 s  (carpetas 1s_X)
         → baseline del estado del arte: división aleatoria sin restricciones.
           Segmentos del mismo archivo original pueden caer en train y test
           al mismo tiempo → métricas optimistas / infladas.

  G-1s : GroupKFold por grabación sobre segmentos de 1 s  (carpetas 1s_X)
         → controlado: todos los segmentos de la misma grabación se asignan
           al mismo fold. Ningún archivo contribuye a train y test
           simultáneamente → estimación más realista del rendimiento.

Los grupos se detectan automáticamente desde los nombres de archivo:
  se elimina el sufijo numérico final (p.ej. _001, -02) del stem asumiendo
  que segmentos del mismo audio original comparten prefijo.

Para cada condición evalúa:
  - Clasificación MULTICLASE : 5 clases originales
                               (asphyxia, deaf, hunger, normal, pain)
  - Clasificación BINARIA    : healthy  (normal + hunger + pain)
                               vs pathology (asphyxia + deaf)

Con los 4 conjuntos de características:
  B: comp+EEFGabor (sin FOSP)
  C: FOSP+comp+EEFGabor (sin reducción dimensional)
  D: FOSP+comp+EEFGabor (SelectKBest k=60 fijo)
  E: FOSP+comp+EEFGabor (Fisher Vector, GMM ajustado por fold)

Análisis estadístico:
  - Intervalos de confianza al 95 % sobre N semillas de CV
  - Prueba de Wilcoxon signed-rank pareada por fold (S-1s vs G-1s)
  - Corrección de Bonferroni (M = 1 par de condiciones por bloque)
  - Tamaño del efecto: Cohen's d

Hiperparámetros: fijos y razonables (mismos que pipeline.py).
El objetivo es la comparación de estrategias de división, no la
optimización de hiperparámetros (para eso ver optuna_search.py /
optuna_binary.py).

Uso:
    python leakage_comparison.py --data_dir "ruta/dataset" \\
        --n_seeds 5 --n_splits 5 --balanced \\
        --csv_results leakage_results.csv \\
        --csv_stats   leakage_stats.csv

    # Solo multiclase, conjuntos B y C
    python leakage_comparison.py --data_dir "ruta/dataset" \\
        --modes multiclass --feature_sets B C --balanced

    # Solo binario, sin conjunto E (Fisher Vector, más lento)
    python leakage_comparison.py --data_dir "ruta/dataset" \\
        --modes binary --feature_sets B C D --balanced
"""

import argparse
import csv
import re
import sys
import time
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, classification_report,
    roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.audio_preprocessing import collect_samples, load_and_preprocess
from src.dataset_analysis import analyze_dataset, print_report
from src.spectral_peaks import extract_spectral_peaks_mel
from src.fosp_surface import extract_fosp_surface_features, fosp_surface_to_vector
from src.fosp_energy_kmeans import (
    extract_fosp_energy_kmeans_features, fosp_energy_kmeans_to_vector, build_mel_bands,
)
from src.complementary_features import extract_complementary_features
from src.eefgabor_features import extract_eefgabor_from_audio
from src.fisher_vector_features import fit_fisher_gmm, encode_fisher_vector

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

FEATURE_SETS = {
    "B": "comp+EEFGabor (sin FOSP)",
    "C": "FOSP+comp+EEFGabor (sin reduc.)",
    "D": "FOSP+comp+EEFGabor (k=60)",
    "E": "FOSP+comp+EEFGabor (FisherVec)",
}

# Condiciones de división: (nombre_corto, descripción, source, tipo_split)
# Ambas usan source="1s" (carpetas 1s_X)
SPLIT_CONDITIONS = [
    ("S-1s", "StratifiedKFold aleatorio — 1s_X (baseline literatura)", "1s", "stratified"),
    ("G-1s", "GroupKFold por grabación — 1s_X (leakage controlado)",   "1s", "group"),
]

HEALTHY_CLASSES   = {"normal", "hunger", "pain"}
PATHOLOGY_CLASSES = {"asphyxia", "deaf"}
BINARY_LABEL_MAP  = {c: "healthy"   for c in HEALTHY_CLASSES}
BINARY_LABEL_MAP |= {c: "pathology" for c in PATHOLOGY_CLASSES}

BOOTSTRAP_B = 2000   # remuestras bootstrap
CI_ALPHA    = 0.05   # nivel de significancia (IC 95 %)


# ---------------------------------------------------------------------------
# Extracción de ID de grabación desde nombre de archivo
# ---------------------------------------------------------------------------

def _recording_id(filepath: str) -> str:
    """
    Extrae el ID de grabación desde el nombre de archivo.

    Esquema confirmado para Baby Chillanto (ver DIAGNOSTICO_RECORDING_ID.md,
    validado con evidencia dura: logs de procesamiento de 2002 encontrados
    en Data/1s_deaf/, coincidencia exacta línea-por-línea/archivo-por-archivo
    para los 6 prefijos observados en esa clase): nombre de archivo de 10
    dígitos PPPPSSSCCC, donde:
      PPPP = ID de grabación (4 dígitos)
      SSS  = contador de segmento dentro de la grabación (3 dígitos,
             reinicia en 001 por cada grabación nueva)
      CCC  = código constante por clase (3 dígitos)

    Ejemplo: 0025001010.wav → grabación "0025" (clase deaf, segmento 001).

    Fallback: si un archivo no sigue el patrón de 10 dígitos (por ejemplo,
    datos de otra fuente añadidos en el futuro), se usa la heurística
    anterior basada en sufijo con separador, para no romper en silencio.
    """
    stem = Path(filepath).stem
    m = re.match(r'^(\d{4})\d{6}$', stem)
    if m:
        return m.group(1)
    cleaned = re.sub(r'[_\-]\d+$', '', stem)
    return cleaned if cleaned != stem else stem


def _assign_groups(paths: list[str]) -> np.ndarray:
    """
    Asigna un entero por grupo de grabación.
    Advierte si todos los grupos son únicos (no se pudo identificar
    agrupación en los nombres de archivo).
    """
    rec_ids = [_recording_id(p) for p in paths]
    unique_ids = sorted(set(rec_ids))
    id_to_int = {uid: i for i, uid in enumerate(unique_ids)}
    groups = np.array([id_to_int[r] for r in rec_ids])

    if len(unique_ids) == len(paths):
        print(
            f"  [AVISO] GroupKFold: no se detectaron grupos en nombres de archivo "
            f"(cada archivo tiene un ID único). GroupKFold degenerará a LOOCV. "
            f"Considera proveer metadatos de grabación."
        )
    else:
        avg_per_group = len(paths) / len(unique_ids)
        print(f"  [INFO] GroupKFold: {len(unique_ids)} grabaciones únicas, "
              f"{avg_per_group:.1f} segmentos/grabación promedio.")
    return groups


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def _load_data(data_dir: str, source: str, fs_target: int, duration_s: float,
               apply_bandpass: bool, low_hz: float, high_hz: float,
               binary: bool = False):
    """
    Carga audios y devuelve (paths, signals, fs, labels_str).
    Si binary=True, remapea etiquetas a {healthy, pathology}.
    """
    raw_samples = collect_samples(data_dir, source=source)
    if not raw_samples:
        raise ValueError(f"No se encontraron muestras en {data_dir} (source='{source}')")

    paths, signals, labels = [], [], []
    for path, clase in raw_samples:
        clase_lower = clase.lower()
        if binary:
            label = BINARY_LABEL_MAP.get(clase_lower)
            if label is None:
                continue
        else:
            label = clase_lower

        try:
            sample = load_and_preprocess(
                path, clase, fs_target=fs_target, duration_s=duration_s,
                apply_bandpass=apply_bandpass, low_hz=low_hz, high_hz=high_hz,
            )
            paths.append(path)
            signals.append(sample.signal)
            labels.append(label)
        except Exception as e:
            print(f"  [ERROR] {path}: {e}")

    return paths, signals, fs_target, np.array(labels)


# ---------------------------------------------------------------------------
# Extracción de características vectoriales
# ---------------------------------------------------------------------------

def _extract_vector_features(signals, fs, n_mels, n_fft, k, sub_window,
                              n_mfcc, f0_fmin, f0_fmax,
                              use_fosp, use_complementary, use_eefgabor):
    fmax = fs / 2.0
    X, feat_names = [], None
    for signal in signals:
        parts, names = [], []
        if use_fosp:
            peaks = extract_spectral_peaks_mel(signal, fs, n_mels=n_mels, n_fft=n_fft,
                                               fmin=0.0, fmax=fmax)
            surf_feat = extract_fosp_surface_features(peaks)
            surf_vec  = fosp_surface_to_vector(surf_feat)
            surf_names = (
                ["fosp_volume_full", "fosp_surface_area_peaks", "fosp_total_energy",
                 "fosp_freq_mean", "fosp_freq_std", "fosp_freq_min", "fosp_freq_max"]
                + [f"fosp_lbp_bin_{i}" for i in range(len(surf_vec) - 7)]
            )
            mel_bands  = build_mel_bands(fmin=0.0, fmax=fmax, n_bands=5)
            energy_feat = extract_fosp_energy_kmeans_features(peaks, k=k,
                           sub_window=sub_window, bands=mel_bands)
            energy_vec  = fosp_energy_kmeans_to_vector(energy_feat)
            energy_names = (
                [f"fosp_centroid_energy_{i}" for i in range(k)]
                + [f"fosp_centroid_freq_{i}"  for i in range(k)]
                + [f"fosp_band_occupancy_{i}" for i in range(5)]
                + [f"fosp_ltp_pos_{i}" for i in range(len(energy_feat.ltp_pos_hist))]
                + [f"fosp_ltp_neg_{i}" for i in range(len(energy_feat.ltp_neg_hist))]
            )
            parts.append(np.concatenate([surf_vec, energy_vec]))
            names += surf_names + energy_names

        if use_complementary:
            comp_vec, comp_names = extract_complementary_features(
                signal, fs, n_mfcc=n_mfcc, n_fft=n_fft,
                f0_fmin=f0_fmin, f0_fmax=f0_fmax)
            parts.append(comp_vec)
            names += comp_names

        if use_eefgabor:
            gabor_vec, gabor_names = extract_eefgabor_from_audio(
                signal, fs, n_mels=n_mels, n_fft=n_fft, fmin=0.0, fmax=fmax)
            parts.append(gabor_vec)
            names += gabor_names

        X.append(np.concatenate(parts) if parts else np.array([]))
        if feat_names is None:
            feat_names = names

    return np.array(X), (feat_names or [])


# ---------------------------------------------------------------------------
# Clasificadores con hiperparámetros fijos
# ---------------------------------------------------------------------------

def _make_classifiers(balanced: bool, random_state: int) -> dict:
    cw = "balanced" if balanced else None
    return {
        "MLP":           MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000,
                                        random_state=random_state),
        "SVM-lineal":    SVC(kernel="linear", class_weight=cw, probability=True),
        "SVM-RBF":       SVC(kernel="rbf",    class_weight=cw, probability=True),
        "SVM-polinomial":SVC(kernel="poly", degree=3, class_weight=cw, probability=True),
        "kNN":           KNeighborsClassifier(n_neighbors=5),
        "RandomForest":  RandomForestClassifier(n_estimators=200, max_depth=15,
                                                class_weight=cw,
                                                random_state=random_state, n_jobs=-1),
    }


def _safe_smote_k(y_enc: np.ndarray, n_train: int) -> tuple[int, bool]:
    """Devuelve (k_neighbors, puede_usar_smote)."""
    min_class_train = int(np.min(np.bincount(
        y_enc[np.random.choice(len(y_enc), n_train, replace=False)]
        if n_train < len(y_enc) else y_enc
    )))
    k = max(1, min(5, min_class_train - 1))
    return k, (min_class_train >= 2)


# ---------------------------------------------------------------------------
# Evaluación de un clasificador con una estrategia de división
# ---------------------------------------------------------------------------

def _evaluate_one(
    X: np.ndarray | None,
    signals: list | None,
    fs: int,
    y_enc: np.ndarray,
    groups: np.ndarray | None,
    le: LabelEncoder,
    clf_name: str,
    clf,
    split_type: str,
    n_splits: int,
    seed: int,
    balanced: bool,
    k_best: int | None,
    feature_set: str,
    n_mels: int,
    n_fft: int,
    fv_n_components: int,
    binary: bool,
) -> dict:
    """
    Ejecuta CV para un clasificador y devuelve métricas por fold.
    Soporta vectorial (X no None) y Fisher Vector (signals no None, X None).
    """
    min_class = int(np.min(np.bincount(y_enc)))
    n_splits_eff = max(2, min(n_splits, min_class))

    if split_type == "stratified":
        cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)
        split_iter = cv.split(
            X if X is not None else np.zeros(len(y_enc)), y_enc)
    else:  # group
        n_groups = len(np.unique(groups))
        n_splits_eff = min(n_splits_eff, n_groups)
        cv = GroupKFold(n_splits=n_splits_eff)
        split_iter = cv.split(
            X if X is not None else np.zeros(len(y_enc)), y_enc, groups)

    fold_kappas = []
    y_true_all, y_pred_all, y_prob_all = [], [], []

    for train_idx, val_idx in split_iter:
        y_tr, y_val = y_enc[train_idx], y_enc[val_idx]

        # ── Fisher Vector: ajustar GMM dentro del fold ──────────────────
        if feature_set == "E":
            try:
                train_sfs = [(signals[i], fs) for i in train_idx]
                fv_gmm = fit_fisher_gmm(train_sfs, n_components=fv_n_components,
                                         n_mels=n_mels, n_fft=n_fft, random_state=seed)
                X_tr  = np.array([encode_fisher_vector(signals[i], fs, fv_gmm, n_fft=n_fft)
                                   for i in train_idx])
                X_val = np.array([encode_fisher_vector(signals[i], fs, fv_gmm, n_fft=n_fft)
                                   for i in val_idx])
            except Exception as e:
                print(f"    [Fisher fold error] {e}")
                continue
        else:
            X_tr, X_val = X[train_idx], X[val_idx]

        # ── Pipeline: scaler + var + (kbest) + (smote) + clf ────────────
        min_tr = int(np.min(np.bincount(y_tr)))
        smote_k = max(1, min(5, min_tr - 1))
        can_smote = (min_tr >= 2) and balanced

        steps = [
            ("scaler", StandardScaler()),
            ("var",    VarianceThreshold(threshold=1e-12)),
        ]
        if k_best is not None and feature_set != "E":
            k_eff = min(k_best, X_tr.shape[1])
            steps.append(("select", SelectKBest(score_func=f_classif, k=k_eff)))
        if can_smote:
            steps.append(("smote", SMOTE(k_neighbors=smote_k, random_state=seed)))
        steps.append(("clf", clf))

        pipe = ImbPipeline(steps)
        try:
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_val)
        except Exception as e:
            print(f"    [fold error {clf_name}] {e}")
            continue

        y_pred_all.extend(preds.tolist())
        y_true_all.extend(y_val.tolist())
        fold_kappas.append(cohen_kappa_score(y_val, preds))

        if hasattr(pipe, "predict_proba"):
            try:
                proba = pipe.predict_proba(X_val)
                y_prob_all.extend(proba.tolist())
            except Exception:
                y_prob_all.extend([[0.5] * len(np.unique(y_enc))] * len(val_idx))
        else:
            y_prob_all.extend([[0.5] * len(np.unique(y_enc))] * len(val_idx))

    if not y_true_all:
        return {}

    yt  = np.array(y_true_all)
    yp  = np.array(y_pred_all)
    ypr = np.array(y_prob_all) if y_prob_all else None

    kappa = cohen_kappa_score(yt, yp)
    acc   = accuracy_score(yt, yp)
    rpt   = classification_report(yt, yp, target_names=le.classes_,
                                   zero_division=0, output_dict=True)

    result = {
        "kappa": kappa,
        "acc":   acc,
        "report": rpt,
        "fold_kappas": fold_kappas,
        "y_true": yt,
        "y_pred": yp,
        "y_prob": np.array(y_prob_all) if y_prob_all else None,
    }

    if binary:
        pathology_idx = list(le.classes_).index("pathology")
        pos = pathology_idx
        cm  = confusion_matrix(yt, yp, labels=[0, 1])
        if cm.shape == (2, 2):
            tp = cm[pos, pos]
            fn = cm[pos, 1 - pos]
            fp = cm[1 - pos, pos]
            tn = cm[1 - pos, 1 - pos]
            result["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            result["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            result["sensitivity"] = 0.0
            result["specificity"] = 0.0

        if ypr is not None and ypr.ndim == 2 and ypr.shape[1] > pos:
            try:
                result["auc"] = roc_auc_score(yt == pos, ypr[:, pos])
            except Exception:
                result["auc"] = 0.0
        else:
            result["auc"] = 0.0

    return result


# ---------------------------------------------------------------------------
# Bootstrap CI al 95 %
# ---------------------------------------------------------------------------

def _bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                   metric_fn, n_boot: int = BOOTSTRAP_B,
                   alpha: float = CI_ALPHA, rng_seed: int = 0) -> tuple[float, float]:
    """
    IC percentil bootstrap al (1-alpha)*100 % para una métrica escalar.
    metric_fn(y_true, y_pred) → float
    """
    rng = np.random.default_rng(rng_seed)
    n = len(y_true)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]
        # Ignorar bootstrap samples con una sola clase (métrica indefinida)
        if len(np.unique(yt_b)) < 2 or len(np.unique(yp_b)) < 1:
            continue
        try:
            boot_vals.append(metric_fn(yt_b, yp_b))
        except Exception:
            pass
    if len(boot_vals) < 10:
        return (float("nan"), float("nan"))
    arr = np.array(boot_vals)
    lo  = float(np.percentile(arr, 100 * alpha / 2))
    hi  = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------------------------------------------------------------------
# Pruebas estadísticas entre condiciones
# ---------------------------------------------------------------------------

def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d: diferencia de medias / std poolada."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                          / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _pairwise_tests(condition_kappas: dict[str, list[float]],
                     bonferroni_m: int = 3) -> list[dict]:
    """
    Wilcoxon signed-rank entre cada par de condiciones.
    Aplica corrección de Bonferroni multiplicando p-valor × bonferroni_m.
    condition_kappas: {condicion: [kappa_fold1, kappa_fold2, ...]}
    """
    rows = []
    cond_names = list(condition_kappas.keys())
    for ca, cb in combinations(cond_names, 2):
        ka = np.array(condition_kappas[ca])
        kb = np.array(condition_kappas[cb])
        # Alinear longitudes (puede diferir si algunos folds fallaron)
        n = min(len(ka), len(kb))
        if n < 4:
            rows.append({
                "cond_A": ca, "cond_B": cb,
                "p_raw": float("nan"), "p_bonf": float("nan"),
                "cohens_d": float("nan"), "significant": False,
            })
            continue
        ka, kb = ka[:n], kb[:n]
        try:
            _, p_raw = stats.wilcoxon(ka, kb, alternative="two-sided")
        except Exception:
            p_raw = float("nan")
        p_bonf = min(1.0, p_raw * bonferroni_m) if not np.isnan(p_raw) else float("nan")
        d = _cohen_d(ka, kb)
        rows.append({
            "cond_A": ca, "cond_B": cb,
            "p_raw":  round(float(p_raw), 6),
            "p_bonf": round(float(p_bonf), 6),
            "cohens_d": round(d, 4),
            "significant": (not np.isnan(p_bonf)) and (p_bonf < CI_ALPHA),
        })
    return rows


# ---------------------------------------------------------------------------
# Bucle principal de evaluación
# ---------------------------------------------------------------------------

def run_comparison(
    mode: str,                  # "multiclass" | "binary"
    feature_set: str,           # "B" | "C" | "D" | "E"
    data_dir: str,
    fs_target: int,
    duration_s: float,
    apply_bandpass: bool,
    low_hz: float,
    high_hz: float,
    n_mels: int,
    n_fft: int,
    k: int,
    sub_window: int,
    n_mfcc: int,
    f0_fmin: float,
    f0_fmax: float,
    n_splits: int,
    balanced: bool,
    n_seeds: int,
    seed_base: int,
    fv_n_components: int,
    k_best_d: int,
    conditions: list[tuple],    # subset de SPLIT_CONDITIONS
    classifiers_to_run: list[str],
) -> tuple[list[dict], list[dict]]:
    """
    Ejecuta las condiciones de división para una (mode, feature_set) dada.
    Devuelve (result_rows, stat_rows).
    """
    binary = (mode == "binary")
    fs_desc  = FEATURE_SETS[feature_set]
    k_best   = k_best_d if feature_set == "D" else None

    fs_cfg = {
        "B": dict(use_fosp=False, use_complementary=True, use_eefgabor=True),
        "C": dict(use_fosp=True,  use_complementary=True, use_eefgabor=True),
        "D": dict(use_fosp=True,  use_complementary=True, use_eefgabor=True),
        "E": None,
    }

    # ── Cargar datos (siempre source=1s) ─────────────────────────────────
    print(f"    Cargando carpetas 1s_X...")
    paths, signals, fs, labels = _load_data(
        data_dir, "1s", fs_target, duration_s,
        apply_bandpass, low_hz, high_hz, binary=binary,
    )
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    data_1s = {"paths": paths, "signals": signals,
               "fs": fs, "labels": labels, "y_enc": y_enc, "le": le}
    # Alias para compatibilidad con el bucle de condiciones
    data_by_source = {"1s": data_1s}
    print(f"    {len(signals)} muestras, clases: {list(le.classes_)}")

    # ── Extraer características vectoriales (una sola vez) ───────────────
    feat_by_source = {}
    if feature_set != "E":
        print(f"    Extrayendo características [{feature_set}] (1s_X)...")
        X, _ = _extract_vector_features(
            signals, fs, n_mels, n_fft, k, sub_window,
            n_mfcc, f0_fmin, f0_fmax, **fs_cfg[feature_set],
        )
        feat_by_source["1s"] = X
        print(f"    {X.shape[1]} características extraídas.")

    result_rows = []
    # Para pruebas estadísticas: acumular kappas por fold por (clf, cond)
    # { clf_name: { cond_name: [kappa_fold, ...] } }
    kappas_for_stats = defaultdict(lambda: defaultdict(list))

    for cond_name, cond_desc, src, split_type in conditions:
        d     = data_by_source[src]
        X_src = feat_by_source.get(src)
        groups = _assign_groups(d["paths"]) if split_type == "group" else None

        for clf_name in classifiers_to_run:
            # Acumular resultados sobre N semillas
            seed_kappas, seed_accs = [], []
            seed_fold_kappas = []
            seed_sens, seed_spec, seed_auc = [], [], []
            seed_reports = []
            # Predicciones acumuladas de todas las semillas (para CM y ROC)
            all_y_true, all_y_pred, all_y_prob = [], [], []

            for seed in range(seed_base, seed_base + n_seeds):
                clf = _make_classifiers(balanced, seed)[clf_name]
                res = _evaluate_one(
                    X=X_src,
                    signals=d["signals"] if feature_set == "E" else None,
                    fs=d["fs"],
                    y_enc=d["y_enc"],
                    groups=groups,
                    le=d["le"],
                    clf_name=clf_name,
                    clf=clf,
                    split_type=split_type,
                    n_splits=n_splits,
                    seed=seed,
                    balanced=balanced,
                    k_best=k_best,
                    feature_set=feature_set,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    fv_n_components=fv_n_components,
                    binary=binary,
                )
                if not res:
                    continue
                seed_kappas.append(res["kappa"])
                seed_accs.append(res["acc"])
                seed_fold_kappas.extend(res["fold_kappas"])
                seed_reports.append(res["report"])
                all_y_true.append(res["y_true"])
                all_y_pred.append(res["y_pred"])
                if res.get("y_prob") is not None:
                    all_y_prob.append(res["y_prob"])
                if binary:
                    seed_sens.append(res.get("sensitivity", 0.0))
                    seed_spec.append(res.get("specificity", 0.0))
                    seed_auc.append(res.get("auc", 0.0))

            if not seed_kappas:
                continue

            kappas_for_stats[clf_name][cond_name].extend(seed_fold_kappas)

            # Bootstrap CI sobre las predicciones de la última semilla
            # (o sobre los kappas por semilla si solo hay 1 fold)
            kappa_arr = np.array(seed_kappas)
            acc_arr   = np.array(seed_accs)

            # CI simple via percentil sobre semillas (rápido)
            def _ci_from_arr(arr):
                if len(arr) < 2:
                    return float("nan"), float("nan")
                lo = float(np.percentile(arr, 2.5))
                hi = float(np.percentile(arr, 97.5))
                return lo, hi

            kappa_ci_lo, kappa_ci_hi = _ci_from_arr(kappa_arr)
            acc_ci_lo,   acc_ci_hi   = _ci_from_arr(acc_arr)

            # F1 por clase
            le_ = d["le"]
            f1_per_class = {}
            for cls in le_.classes_:
                vals = [rpt.get(cls, {}).get("f1-score", 0.0) for rpt in seed_reports]
                f1_per_class[cls] = (float(np.mean(vals)), float(np.std(vals)))

            row = {
                "mode":             mode,
                "feature_set":      feature_set,
                "feature_set_desc": fs_desc,
                "condition":        cond_name,
                "condition_desc":   cond_desc,
                "classifier":       clf_name,
                "n_seeds":          len(seed_kappas),
                "kappa_mean":       round(float(np.mean(kappa_arr)), 4),
                "kappa_std":        round(float(np.std(kappa_arr)),  4),
                "kappa_ci_lo":      round(kappa_ci_lo, 4),
                "kappa_ci_hi":      round(kappa_ci_hi, 4),
                "acc_mean":         round(float(np.mean(acc_arr)),   4),
                "acc_std":          round(float(np.std(acc_arr)),    4),
                "acc_ci_lo":        round(acc_ci_lo, 4),
                "acc_ci_hi":        round(acc_ci_hi, 4),
            }
            if binary:
                for key, arr in [("sensitivity", seed_sens),
                                  ("specificity", seed_spec),
                                  ("auc",         seed_auc)]:
                    a = np.array(arr)
                    lo_, hi_ = _ci_from_arr(a)
                    row[f"{key}_mean"] = round(float(np.mean(a)), 4)
                    row[f"{key}_std"]  = round(float(np.std(a)),  4)
                    row[f"{key}_ci_lo"] = round(lo_, 4)
                    row[f"{key}_ci_hi"] = round(hi_, 4)

            for cls, (f1m, f1s) in f1_per_class.items():
                row[f"f1_{cls}_mean"] = round(f1m, 4)
                row[f"f1_{cls}_std"]  = round(f1s, 4)

            # Predicciones concatenadas de todas las semillas (para CM y ROC)
            row["_y_true"]  = np.concatenate(all_y_true)  if all_y_true  else np.array([])
            row["_y_pred"]  = np.concatenate(all_y_pred)  if all_y_pred  else np.array([])
            row["_y_prob"]  = np.concatenate(all_y_prob)  if all_y_prob  else None
            row["_classes"] = list(d["le"].classes_)

            result_rows.append(row)

    # ── Pruebas estadísticas entre condiciones ────────────────────────────
    stat_rows = []
    n_comparisons = 1  # solo 1 par: S-1s vs G-1s
    for clf_name in classifiers_to_run:
        tests = _pairwise_tests(kappas_for_stats[clf_name],
                                 bonferroni_m=n_comparisons)
        for t in tests:
            stat_rows.append({
                "mode":         mode,
                "feature_set":  feature_set,
                "classifier":   clf_name,
                **t,
            })

    return result_rows, stat_rows


# ---------------------------------------------------------------------------
# Salida por pantalla
# ---------------------------------------------------------------------------

def _print_block(title: str, rows: list[dict], binary: bool):
    if not rows:
        return
    print(f"\n{'='*115}")
    print(f"  {title}")
    print(f"{'='*115}")

    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["feature_set"], r["condition"])].append(r)

    for fs_key in ["B", "C", "D", "E"]:
        for cond_name, _, _, _ in SPLIT_CONDITIONS:
            block = grouped.get((fs_key, cond_name), [])
            if not block:
                continue
            cond_desc = block[0]["condition_desc"]
            fs_desc   = block[0]["feature_set_desc"]
            print(f"\n  [{fs_key}] {fs_desc}  ─  {cond_name}: {cond_desc}")

            if binary:
                hdr = (f"    {'Clasificador':<16}{'Kappa':>8}{'±':>5}"
                       f"{'IC95%':>14}{'Acc':>8}{'Sens':>8}{'Spec':>8}{'AUC':>8}")
            else:
                hdr = (f"    {'Clasificador':<16}{'Kappa':>8}{'±':>5}"
                       f"{'IC95%':>14}{'Acc':>8}{'Acc_std':>9}")
            print(hdr)
            print("    " + "-" * (len(hdr) - 4))

            for r in sorted(block, key=lambda x: x["kappa_mean"], reverse=True):
                ic = f"[{r['kappa_ci_lo']:.3f},{r['kappa_ci_hi']:.3f}]"
                if binary:
                    print(f"    {r['classifier']:<16}"
                          f"{r['kappa_mean']:>8.4f}{r['kappa_std']:>5.3f}"
                          f"{ic:>14}"
                          f"{r['acc_mean']:>8.4f}"
                          f"{r.get('sensitivity_mean',0):>8.4f}"
                          f"{r.get('specificity_mean',0):>8.4f}"
                          f"{r.get('auc_mean',0):>8.4f}")
                else:
                    print(f"    {r['classifier']:<16}"
                          f"{r['kappa_mean']:>8.4f}{r['kappa_std']:>5.3f}"
                          f"{ic:>14}"
                          f"{r['acc_mean']:>8.4f}{r['acc_std']:>9.4f}")


def _print_normalized_cm(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list[str], title: str):
    """Imprime una matriz de confusión normalizada por fila (recall por clase)."""
    if len(y_true) == 0:
        return
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Normalizar por fila (verdadero positivo / total real de esa clase)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1          # evitar división por cero
    cm_norm = cm.astype(float) / row_sums

    col_w = 9
    name_w = max(12, max(len(c) for c in class_names) + 1)

    print(f"\n  ── {title}")
    print(f"  {'(normalizada por fila — recall por clase)'}")

    # Cabecera
    header = f"  {'Verdadero↓ / Pred→':<{name_w}}" + "".join(
        f"{c[:col_w-1]:>{col_w}}" for c in class_names)
    print(header)
    print("  " + "─" * len(header))

    # Filas
    for i, cls in enumerate(class_names):
        row_str = f"  {cls:<{name_w}}"
        for j in range(len(class_names)):
            val = cm_norm[i, j]
            # Resaltar diagonal
            cell = f"{val:.2f}"
            if i == j:
                cell = f"[{val:.2f}]"
            row_str += f"{cell:>{col_w}}"
        row_str += f"   (n={cm[i].sum()})"
        print(row_str)
    print()


def _plot_confusion_matrices(all_result_rows: list[dict], plot_dir: str):
    """
    Para cada conjunto de características, busca el MEJOR clasificador multiclase
    (mayor kappa_mean en S-1s) y genera una figura con dos matrices de confusión
    normalizadas en paralelo: S-1s (izquierda) y G-1s (derecha).
    Guarda PNG en plot_dir.
    """
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    mc_rows = [r for r in all_result_rows if r["mode"] == "multiclass"]
    if not mc_rows:
        return

    for fs_key in ["B", "C", "D", "E"]:
        fs_rows = [r for r in mc_rows if r["feature_set"] == fs_key]
        if not fs_rows:
            continue

        # Mejor clasificador según kappa_mean en S-1s
        s1s = {r["classifier"]: r for r in fs_rows if r["condition"] == "S-1s"}
        g1s = {r["classifier"]: r for r in fs_rows if r["condition"] == "G-1s"}
        if not s1s or not g1s:
            continue
        best_clf = max(s1s, key=lambda c: s1s[c]["kappa_mean"])

        row_s = s1s[best_clf]
        row_g = g1s.get(best_clf)
        if row_g is None:
            continue

        class_names = row_s["_classes"]
        n_classes   = len(class_names)

        def _norm_cm(yt, yp):
            cm = confusion_matrix(yt, yp, labels=list(range(n_classes)))
            rs = cm.sum(axis=1, keepdims=True)
            rs[rs == 0] = 1
            return cm.astype(float) / rs, cm

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        pairs = [
            (axes[0], row_s, "S-1s — StratifiedKFold (baseline)"),
            (axes[1], row_g, "G-1s — GroupKFold (controlado)"),
        ]

        for ax, row, title in pairs:
            yt = row["_y_true"]
            yp = row["_y_pred"]
            if len(yt) == 0:
                ax.set_visible(False)
                continue
            cm_norm, cm_raw = _norm_cm(yt, yp)
            im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
                           vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set(xticks=range(n_classes), yticks=range(n_classes),
                   xticklabels=class_names, yticklabels=class_names)
            ax.set_xlabel("Predicho", fontsize=10)
            ax.set_ylabel("Real", fontsize=10)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
            plt.setp(ax.get_yticklabels(), fontsize=8)
            kappa = row["kappa_mean"]
            ax.set_title(f"{title}\n{best_clf}  |  κ = {kappa:.3f}", fontsize=9)
            thresh = 0.5
            for i in range(n_classes):
                for j in range(n_classes):
                    val  = cm_norm[i, j]
                    text = f"{val:.2f}\n(n={cm_raw[i,j]})"
                    color = "white" if val > thresh else "black"
                    ax.text(j, i, text, ha="center", va="center",
                            fontsize=7, color=color)

        fs_desc = FEATURE_SETS[fs_key].replace(" ", "_").replace("/", "-")
        fig.suptitle(
            f"Matrices de confusión normalizadas — Multiclase\n"
            f"Conjunto {fs_key}: {FEATURE_SETS[fs_key]}",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout()
        out = Path(plot_dir) / f"cm_multiclass_{fs_key}_{best_clf.replace('-','_')}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Matriz de confusión guardada: {out}")


def _plot_roc_curves(all_result_rows: list[dict], plot_dir: str):
    """
    Para cada conjunto de características, genera una figura con las curvas
    ROC de todos los clasificadores bajo las dos condiciones (S-1s y G-1s),
    incluyendo el AUC en la leyenda. Solo aplica a modo binario.
    Guarda PNG en plot_dir.
    """
    from sklearn.metrics import roc_curve
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    bin_rows = [r for r in all_result_rows if r["mode"] == "binary"]
    if not bin_rows:
        return

    # Índice de la clase positiva 'pathology' (igual en todas las rows)
    sample_classes = bin_rows[0]["_classes"]
    pos_idx = sample_classes.index("pathology") if "pathology" in sample_classes else 1

    cond_styles = {
        "S-1s": {"ls": "--", "alpha": 0.75},
        "G-1s": {"ls": "-",  "alpha": 1.00},
    }
    cmap = plt.cm.get_cmap("tab10")

    for fs_key in ["B", "C", "D", "E"]:
        fs_rows = [r for r in bin_rows if r["feature_set"] == fs_key]
        if not fs_rows:
            continue

        classifiers = sorted({r["classifier"] for r in fs_rows})
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        ax_map = {"S-1s": axes[0], "G-1s": axes[1]}

        for ci, clf_name in enumerate(classifiers):
            color = cmap(ci % 10)
            for cond_name, cond_style in cond_styles.items():
                row = next((r for r in fs_rows
                            if r["classifier"] == clf_name
                            and r["condition"] == cond_name), None)
                if row is None:
                    continue
                yt   = row["_y_true"]
                yprob = row["_y_prob"]
                if len(yt) == 0 or yprob is None:
                    continue
                if yprob.ndim == 2 and yprob.shape[1] > pos_idx:
                    scores = yprob[:, pos_idx]
                else:
                    continue
                try:
                    fpr, tpr, _ = roc_curve(yt == pos_idx, scores)
                    auc_val = float(roc_auc_score(yt == pos_idx, scores))
                except Exception:
                    continue

                ax = ax_map[cond_name]
                ax.plot(fpr, tpr, color=color,
                        ls=cond_style["ls"], alpha=cond_style["alpha"],
                        lw=1.6,
                        label=f"{clf_name}  AUC={auc_val:.3f}")

        for cond_name, ax in ax_map.items():
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Azar")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.set_xlabel("Tasa de falsos positivos (1 − Especificidad)", fontsize=9)
            ax.set_ylabel("Tasa de verdaderos positivos (Sensibilidad)", fontsize=9)
            ax.set_title(f"{cond_name}", fontsize=10, fontweight="bold")
            ax.legend(fontsize=7, loc="lower right")
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

        fig.suptitle(
            f"Curvas ROC — Clasificación binaria (healthy vs. pathology)\n"
            f"Conjunto {fs_key}: {FEATURE_SETS[fs_key]}",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout()
        out = Path(plot_dir) / f"roc_binary_{fs_key}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Curvas ROC guardadas: {out}")


def _print_stats(stat_rows: list[dict]):
    if not stat_rows:
        return
    print(f"\n{'='*100}")
    print("  PRUEBAS DE SIGNIFICANCIA ESTADÍSTICA (Wilcoxon signed-rank + Corrección Bonferroni)")
    print(f"  α = {CI_ALPHA}  |  M = {len(stat_rows)} comparaciones (familywise, corrección sobre el total)")
    print(f"{'='*100}")
    hdr = (f"  {'Modo':<12}{'Set':<5}{'Clasificador':<16}"
           f"{'Cond_A':<8}{'Cond_B':<8}{'p_raw':>9}{'p_bonf':>9}"
           f"{'Cohen_d':>9}{'Sig':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in stat_rows:
        sig_mark = "  ✓" if r["significant"] else ""
        print(f"  {r['mode']:<12}{r['feature_set']:<5}{r['classifier']:<16}"
              f"{r['cond_A']:<8}{r['cond_B']:<8}"
              f"{r['p_raw']:>9.4f}{r['p_bonf']:>9.4f}"
              f"{r['cohens_d']:>9.4f}{sig_mark:>6}")


def _print_leakage_summary(all_result_rows: list[dict]):
    """Tabla resumen del efecto de leakage: Kappa S-1s vs G-1s por mejor clf."""
    print(f"\n{'='*90}")
    print("  RESUMEN DE INFLACIÓN POR LEAKAGE: S-1s vs G-1s (mejor clasificador por bloque)")
    print(f"  Δ Kappa positivo = inflación (literatura optimista vs. controlado por grabación)")
    print(f"{'='*90}")
    hdr = f"  {'Modo':<12}{'Set':<5}{'Clasificador':<16}{'S-1s':>8}{'G-1s':>8}{'Δ Kappa':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for mode in ["multiclass", "binary"]:
        for fs_key in ["B", "C", "D", "E"]:
            block = [r for r in all_result_rows
                     if r["mode"] == mode and r["feature_set"] == fs_key]
            if not block:
                continue
            # Mejor clf según Kappa en S-1s
            s1s_rows = {r["classifier"]: r for r in block if r["condition"] == "S-1s"}
            g1s_rows = {r["classifier"]: r for r in block if r["condition"] == "G-1s"}
            if not s1s_rows or not g1s_rows:
                continue
            best_clf = max(s1s_rows, key=lambda c: s1s_rows[c]["kappa_mean"])
            k_s1s = s1s_rows[best_clf]["kappa_mean"]
            k_g1s = g1s_rows.get(best_clf, {}).get("kappa_mean", float("nan"))
            delta  = k_s1s - k_g1s if not np.isnan(k_g1s) else float("nan")
            print(f"  {mode:<12}{fs_key:<5}{best_clf:<16}"
                  f"{k_s1s:>8.4f}{k_g1s:>8.4f}{delta:>9.4f}")


# ---------------------------------------------------------------------------
# Guardado en CSV
# ---------------------------------------------------------------------------

def _save_csv(rows: list[dict], path: str):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        print(f"[INFO] CSV vacío guardado en: {path}")
        return

    # Excluir columnas internas (arrays numpy no serializables en CSV)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    all_keys = list(clean[0].keys())
    for r in clean[1:]:
        for k in r.keys():
            if k not in all_keys:
                all_keys.append(k)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in clean:
            w.writerow(r)
    print(f"[INFO] CSV guardado en: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Comparación de leakage: StratifiedKFold vs GroupKFold por grabación"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--duration_s",   type=float, default=1.0)
    parser.add_argument("--fs_target",    type=int,   default=16000)
    parser.add_argument("--no_bandpass",  action="store_true")
    parser.add_argument("--low_hz",       type=float, default=150.0)
    parser.add_argument("--high_hz",      type=float, default=8000.0)
    parser.add_argument("--n_mels",       type=int,   default=40)
    parser.add_argument("--n_fft",        type=int,   default=1024)
    parser.add_argument("--k",            type=int,   default=5)
    parser.add_argument("--sub_window",   type=int,   default=4)
    parser.add_argument("--n_mfcc",       type=int,   default=20)
    parser.add_argument("--f0_fmin",      type=float, default=80.0)
    parser.add_argument("--f0_fmax",      type=float, default=1000.0)
    parser.add_argument("--k_best_d",     type=int,   default=60,
                        help="k fijo para conjunto D")
    parser.add_argument("--fv_n_components", type=int, default=16,
                        help="Componentes GMM para Fisher Vector")
    parser.add_argument("--feature_sets", nargs="+", default=["B", "C", "D", "E"],
                        choices=["B", "C", "D", "E"])
    parser.add_argument("--modes", nargs="+", default=["multiclass", "binary"],
                        choices=["multiclass", "binary"])
    parser.add_argument("--conditions", nargs="+", default=["S-1s", "G-1s"],
                        choices=["S-1s", "G-1s"],
                        help="Condiciones de división a evaluar (ambas usan 1s_X)")
    parser.add_argument("--classifiers", nargs="+",
                        default=["MLP", "SVM-lineal", "SVM-RBF",
                                 "SVM-polinomial", "kNN", "RandomForest"])
    parser.add_argument("--n_splits",  type=int, default=5)
    parser.add_argument("--n_seeds",   type=int, default=5,
                        help="Semillas para estabilidad estadística")
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--balanced",  action="store_true")
    parser.add_argument("--skip_analysis", action="store_true")
    parser.add_argument("--csv_results", default="leakage_results.csv")
    parser.add_argument("--csv_stats",   default="leakage_stats.csv")
    parser.add_argument("--plot_dir",    default="leakage_plots",
                        help="Carpeta donde guardar las figuras PNG (CM y ROC)")
    args = parser.parse_args()

    if not args.skip_analysis:
        print(f"\nAnalizando dataset (carpetas 1s_X)...")
        try:
            report = analyze_dataset(args.data_dir, source="1s")
            print_report(report)
        except Exception as e:
            print(f"[AVISO] Análisis omitido: {e}")

    # Filtrar condiciones seleccionadas
    selected_conditions = [c for c in SPLIT_CONDITIONS if c[0] in args.conditions]

    all_results, all_stats = [], []

    try:
        for mode in args.modes:
            mode_label = "MULTICLASE (5 clases)" if mode == "multiclass" \
                         else "BINARIO (healthy vs pathology)"
            print(f"\n{'#'*70}")
            print(f"  MODO: {mode_label}")
            if mode == "binary":
                print(f"  healthy   ← {sorted(HEALTHY_CLASSES)}")
                print(f"  pathology ← {sorted(PATHOLOGY_CLASSES)}")
            print(f"{'#'*70}")

            for fs_key in args.feature_sets:
                print(f"\n  ── Conjunto {fs_key}: {FEATURE_SETS[fs_key]} ──")
                t0 = time.time()

                try:
                    result_rows, stat_rows = run_comparison(
                        mode=mode,
                        feature_set=fs_key,
                        data_dir=args.data_dir,
                        fs_target=args.fs_target,
                        duration_s=args.duration_s,
                        apply_bandpass=not args.no_bandpass,
                        low_hz=args.low_hz,
                        high_hz=args.high_hz,
                        n_mels=args.n_mels,
                        n_fft=args.n_fft,
                        k=args.k,
                        sub_window=args.sub_window,
                        n_mfcc=args.n_mfcc,
                        f0_fmin=args.f0_fmin,
                        f0_fmax=args.f0_fmax,
                        n_splits=args.n_splits,
                        balanced=args.balanced,
                        n_seeds=args.n_seeds,
                        seed_base=args.seed_base,
                        fv_n_components=args.fv_n_components,
                        k_best_d=args.k_best_d,
                        conditions=selected_conditions,
                        classifiers_to_run=args.classifiers,
                    )
                except Exception as exc:
                    print(f"[ERROR] Fallo en {mode}/{fs_key}: {exc}")
                    result_rows, stat_rows = [], []

                all_results.extend(result_rows)
                all_stats.extend(stat_rows)
                print(f"  Completado en {time.time()-t0:.1f}s")

            # Imprimir resultados del modo
            _print_block(
                f"RESULTADOS — {mode_label}",
                [r for r in all_results if r["mode"] == mode],
                binary=(mode == "binary"),
            )

        # Corrección de Bonferroni familywise sobre el total real de pruebas
        m_total = len(all_stats)
        for row in all_stats:
            p_raw = row["p_raw"]
            if p_raw != p_raw:  # NaN
                row["p_bonf"] = float("nan")
                row["significant"] = False
                continue
            p_bonf = min(1.0, p_raw * m_total)
            row["p_bonf"] = round(p_bonf, 6)
            row["significant"] = p_bonf < 0.05

        # Tablas de significancia y resumen de inflación
        _print_stats(all_stats)
        _print_leakage_summary(all_results)

        # Figuras
        print(f"\n[PLOT] Generando figuras en: {args.plot_dir}")
        _plot_confusion_matrices(all_results, args.plot_dir)
        _plot_roc_curves(all_results, args.plot_dir)
    finally:
        # Guardar CSVs (excluir columnas internas _y_true / _y_pred / _y_prob)
        csv_rows = [{k: v for k, v in r.items() if not k.startswith("_")}
                    for r in all_results]
        _save_csv(csv_rows,    args.csv_results)
        _save_csv(all_stats,   args.csv_stats)


if __name__ == "__main__":
    main()
