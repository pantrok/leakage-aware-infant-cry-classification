"""
Clasificación BINARIA de llanto de bebé con Optuna (TPE + ASHA)
================================================================
Agrupa las 5 clases originales en 2 clases de alto nivel:

  healthy   ← normal, hunger, pain   (llanto fisiológico / saludable)
  pathology ← asphyxia, deaf         (llanto patológico)

y evalúa las mismas 4 combinaciones de características que el experimento
multiclase (optuna_search.py):

  B) comp+EEFGabor                    — complementarias + Gabor, sin FOSP
  C) FOSP+comp+EEFGabor (sin reduc.)  — todas las características, sin SelectKBest
  D) FOSP+comp+EEFGabor (k=60)        — igual que C pero SelectKBest k=60 fijo
  E) FOSP+comp+EEFGabor (FisherVec)   — Fisher Vector (GMM ajustado por fold)

Metodología idéntica al experimento multiclase:
  - TPE (Tree-structured Parzen Estimator) + SuccessiveHalvingPruner (ASHA)
  - Validación cruzada estratificada (binaria: healthy vs pathology)
  - Evaluación final con N semillas para media ± std de Kappa
  - Métricas adicionales: AUC-ROC, sensibilidad y especificidad

Uso:
    python optuna_binary.py --data_dir "ruta/dataset" --source 1s \\
        --n_trials 100 --n_seeds 5 --balanced \\
        --csv_summary binary_summary.csv --csv_history binary_history.csv

    # Solo conjuntos B y D
    python optuna_binary.py --data_dir "ruta/dataset" --source 1s \\
        --feature_sets B D --classifiers SVM-RBF kNN --n_trials 80 --balanced
"""

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.metrics import (
    cohen_kappa_score, classification_report, accuracy_score,
    roc_auc_score, roc_curve, auc as sk_auc, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
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
from src.gpu_classifiers import (
    GPU_AVAILABLE, TORCH_CUDA_AVAILABLE, _CUDA_RUNTIME_BROKEN,
    TorchMLPClassifier, report_backend,
)

try:
    from cuml.svm import SVC as cuSVC
    from cuml.neighbors import KNeighborsClassifier as cuKNN
    from cuml.ensemble import RandomForestClassifier as cuRF
    _CUML_OK = GPU_AVAILABLE
except ImportError:
    _CUML_OK = False

from sklearn.svm import SVC as skSVC
from sklearn.neighbors import KNeighborsClassifier as skKNN
from sklearn.neural_network import MLPClassifier as skMLP
from sklearn.ensemble import RandomForestClassifier as skRF

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Mapeo multiclase → binario
# ---------------------------------------------------------------------------

HEALTHY_CLASSES    = {"normal", "hunger", "pain"}
PATHOLOGY_CLASSES  = {"asphyxia", "deaf"}
BINARY_LABEL_MAP   = {c: "healthy"   for c in HEALTHY_CLASSES}
BINARY_LABEL_MAP  |= {c: "pathology" for c in PATHOLOGY_CLASSES}

FEATURE_SETS = {
    "B": "comp+EEFGabor (sin FOSP)",
    "C": "FOSP+comp+EEFGabor (sin reducción)",
    "D": "FOSP+comp+EEFGabor (SelectKBest k=60)",
    "E": "FOSP+comp+EEFGabor (Fisher Vector)",
}


# ---------------------------------------------------------------------------
# Carga de audios y remapeo a etiquetas binarias
# ---------------------------------------------------------------------------

def _load_signals_binary(data_dir: str, source: str, fs_target: int, duration_s: float,
                          apply_bandpass: bool, low_hz: float, high_hz: float):
    """
    Carga audios y remapea etiquetas a {healthy, pathology}.
    Clases no reconocidas se descartan con aviso.
    """
    raw_samples = collect_samples(data_dir, source=source)
    if not raw_samples:
        raise ValueError(f"No se encontraron muestras en {data_dir} (source='{source}')")

    signals, labels = [], []
    skipped = {}
    for path, clase in raw_samples:
        clase_lower = clase.lower()
        binary_label = BINARY_LABEL_MAP.get(clase_lower)
        if binary_label is None:
            skipped[clase_lower] = skipped.get(clase_lower, 0) + 1
            continue
        try:
            sample = load_and_preprocess(
                path, clase, fs_target=fs_target, duration_s=duration_s,
                apply_bandpass=apply_bandpass, low_hz=low_hz, high_hz=high_hz,
            )
            signals.append(sample.signal)
            labels.append(binary_label)
        except Exception as e:
            print(f"[ERROR] Fallo al procesar {path}: {e}")

    if skipped:
        print(f"[AVISO] Clases no reconocidas (descartadas): {skipped}")

    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"[INFO] Distribución binaria:")
    for u, c in zip(unique, counts):
        print(f"       {u:<12}: {c} muestras")

    return signals, fs_target, labels


# ---------------------------------------------------------------------------
# Extracción de características vectoriales
# ---------------------------------------------------------------------------

def _extract_vector_features(signals, fs, n_mels, n_fft, k, sub_window,
                              n_mfcc, f0_fmin, f0_fmax,
                              use_fosp: bool, use_complementary: bool, use_eefgabor: bool):
    fmax = fs / 2.0
    X, feature_names = [], None
    for signal in signals:
        parts, names = [], []
        if use_fosp:
            peaks = extract_spectral_peaks_mel(signal, fs, n_mels=n_mels, n_fft=n_fft,
                                               fmin=0.0, fmax=fmax)
            surf_feat = extract_fosp_surface_features(peaks)
            surf_vec = fosp_surface_to_vector(surf_feat)
            surf_names = (
                ["fosp_volume_full", "fosp_surface_area_peaks", "fosp_total_energy",
                 "fosp_freq_mean", "fosp_freq_std", "fosp_freq_min", "fosp_freq_max"]
                + [f"fosp_lbp_bin_{i}" for i in range(len(surf_vec) - 7)]
            )
            mel_bands = build_mel_bands(fmin=0.0, fmax=fmax, n_bands=5)
            energy_feat = extract_fosp_energy_kmeans_features(
                peaks, k=k, sub_window=sub_window, bands=mel_bands)
            energy_vec = fosp_energy_kmeans_to_vector(energy_feat)
            energy_names = (
                [f"fosp_centroid_energy_{i}" for i in range(k)]
                + [f"fosp_centroid_freq_{i}" for i in range(k)]
                + [f"fosp_band_occupancy_{i}" for i in range(5)]
                + [f"fosp_ltp_pos_{i}" for i in range(len(energy_feat.ltp_pos_hist))]
                + [f"fosp_ltp_neg_{i}" for i in range(len(energy_feat.ltp_neg_hist))]
            )
            parts.append(np.concatenate([surf_vec, energy_vec]))
            names += surf_names + energy_names
        if use_complementary:
            comp_vec, comp_names = extract_complementary_features(
                signal, fs, n_mfcc=n_mfcc, n_fft=n_fft, f0_fmin=f0_fmin, f0_fmax=f0_fmax)
            parts.append(comp_vec)
            names += comp_names
        if use_eefgabor:
            gabor_vec, gabor_names = extract_eefgabor_from_audio(
                signal, fs, n_mels=n_mels, n_fft=n_fft, fmin=0.0, fmax=fmax)
            parts.append(gabor_vec)
            names += gabor_names
        full_vec = np.concatenate(parts) if parts else np.array([])
        X.append(full_vec)
        if feature_names is None:
            feature_names = names
    return np.array(X), (feature_names or [])


# ---------------------------------------------------------------------------
# MLP con poda a nivel de época para GPU/PyTorch
# ---------------------------------------------------------------------------

class PrunableTorchMLP(TorchMLPClassifier):
    def __init__(self, trial=None, fold_step_offset=0, prune_interval=50, **kwargs):
        super().__init__(**kwargs)
        self.trial = trial
        self.fold_step_offset = fold_step_offset
        self.prune_interval = prune_interval

    def fit(self, X, y):
        if self.trial is None or not TORCH_CUDA_AVAILABLE:
            return super().fit(X, y)

        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state)
        device = self._get_device()
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        model = self._build_model(n_features, n_classes, device)

        if device == "cuda":
            try:
                with torch.no_grad():
                    probe = torch.zeros((1, n_features), dtype=torch.float32, device=device)
                    model(probe)
            except RuntimeError as e:
                if not _CUDA_RUNTIME_BROKEN[0]:
                    print(f"[AVISO] GPU no utilizable ({e}). Cayendo a CPU.")
                _CUDA_RUNTIME_BROKEN[0] = True
                device = "cpu"
                model = self._build_model(n_features, n_classes, device)

        self._model = model
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=self.alpha)

        class_weights_tensor = None
        if self.class_weight == "balanced":
            y_idx_for_weights = np.searchsorted(self.classes_, y)
            counts = np.bincount(y_idx_for_weights, minlength=n_classes)
            counts = np.maximum(counts, 1)
            weights = len(y) / (n_classes * counts)
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_idx = np.searchsorted(self.classes_, y)
        y_t = torch.tensor(y_idx, dtype=torch.long, device=device)

        n_epochs = self.max_iter // 10
        self._model.train()
        for ep in range(n_epochs):
            optimizer.zero_grad()
            out = self._model(X_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()
            if self.trial is not None and (ep + 1) % self.prune_interval == 0:
                step = self.fold_step_offset * n_epochs + ep
                self.trial.report(float(loss.item()), step=step)
                if self.trial.should_prune():
                    raise optuna.TrialPruned()

        self._device = device
        return self


# ---------------------------------------------------------------------------
# Construcción de clasificadores
# ---------------------------------------------------------------------------

def _suggest_and_build_classifier(trial, clf_name, balanced, random_state,
                                   fold_step_offset=0):
    cw = "balanced" if balanced else None

    if clf_name == "MLP":
        h1 = trial.suggest_int("mlp_hidden1", 16, 256, log=True)
        h2 = trial.suggest_int("mlp_hidden2", 8, 128, log=True)
        alpha = trial.suggest_float("mlp_alpha", 1e-6, 1e-1, log=True)
        if TORCH_CUDA_AVAILABLE and not _CUDA_RUNTIME_BROKEN[0]:
            lr = trial.suggest_float("mlp_lr", 1e-4, 1e-1, log=True)
            return PrunableTorchMLP(
                trial=trial, fold_step_offset=fold_step_offset,
                hidden1=h1, hidden2=h2, lr=lr, alpha=alpha,
                max_iter=2000, random_state=random_state, class_weight=cw)
        lr_init = trial.suggest_float("mlp_lr", 1e-4, 1e-1, log=True)
        return skMLP(hidden_layer_sizes=(h1, h2), alpha=alpha,
                     learning_rate_init=lr_init, max_iter=2000, random_state=random_state)

    if clf_name == "SVM-lineal":
        C = trial.suggest_float("svm_lin_C", 1e-3, 1e3, log=True)
        if _CUML_OK:
            return cuSVC(kernel="linear", C=C, class_weight=cw)
        return skSVC(kernel="linear", C=C, class_weight=cw, probability=True)

    if clf_name == "SVM-RBF":
        C = trial.suggest_float("svm_rbf_C", 1e-2, 1e4, log=True)
        gamma_mode = trial.suggest_categorical("svm_rbf_gamma_mode",
                                               ["scale", "auto", "value"])
        gamma = (trial.suggest_float("svm_rbf_gamma_val", 1e-5, 10.0, log=True)
                 if gamma_mode == "value" else gamma_mode)
        if _CUML_OK:
            return cuSVC(kernel="rbf", C=C, gamma=gamma, class_weight=cw)
        return skSVC(kernel="rbf", C=C, gamma=gamma, class_weight=cw, probability=True)

    if clf_name == "SVM-polinomial":
        C = trial.suggest_float("svm_poly_C", 1e-2, 1e3, log=True)
        degree = trial.suggest_int("svm_poly_degree", 2, 5)
        coef0 = trial.suggest_float("svm_poly_coef0", 0.0, 3.0)
        if _CUML_OK:
            return cuSVC(kernel="poly", C=C, degree=degree, coef0=coef0, class_weight=cw)
        return skSVC(kernel="poly", C=C, degree=degree, coef0=coef0,
                     class_weight=cw, probability=True)

    if clf_name == "kNN":
        n_neighbors = trial.suggest_int("knn_n_neighbors", 3, 20)
        weights = trial.suggest_categorical("knn_weights", ["uniform", "distance"])
        p = trial.suggest_categorical("knn_p", [1, 2])
        if _CUML_OK:
            return cuKNN(n_neighbors=n_neighbors, weights=weights, p=p)
        return skKNN(n_neighbors=n_neighbors, weights=weights, p=p)

    if clf_name == "RandomForest":
        n_estimators = trial.suggest_int("rf_n_estimators", 50, 500, log=True)
        max_depth = trial.suggest_int("rf_max_depth", 3, 30)
        min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 10)
        max_features = trial.suggest_categorical(
            "rf_max_features", ["sqrt", "log2", 0.3, 0.5, 0.7])
        if _CUML_OK:
            return cuRF(n_estimators=n_estimators, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         random_state=random_state)
        return skRF(n_estimators=n_estimators, max_depth=max_depth,
                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                     max_features=max_features, class_weight=cw,
                     random_state=random_state, n_jobs=-1)

    raise ValueError(f"Clasificador desconocido: {clf_name}")


def _build_clf_from_params(clf_name, params, cw, random_state):
    if clf_name == "MLP":
        h1 = int(params.get("mlp_hidden1", 64))
        h2 = int(params.get("mlp_hidden2", 32))
        alpha = float(params.get("mlp_alpha", 1e-4))
        lr = float(params.get("mlp_lr", 1e-3))
        if TORCH_CUDA_AVAILABLE and not _CUDA_RUNTIME_BROKEN[0]:
            return TorchMLPClassifier(hidden1=h1, hidden2=h2, lr=lr, alpha=alpha,
                                       max_iter=2000, random_state=random_state, class_weight=cw)
        return skMLP(hidden_layer_sizes=(h1, h2), alpha=alpha,
                     learning_rate_init=lr, max_iter=2000, random_state=random_state)
    if clf_name == "SVM-lineal":
        C = float(params.get("svm_lin_C", 1.0))
        if _CUML_OK:
            return cuSVC(kernel="linear", C=C, class_weight=cw)
        return skSVC(kernel="linear", C=C, class_weight=cw, probability=True)
    if clf_name == "SVM-RBF":
        C = float(params.get("svm_rbf_C", 1.0))
        gmode = params.get("svm_rbf_gamma_mode", "scale")
        gamma = float(params.get("svm_rbf_gamma_val", 0.001)) if gmode == "value" else gmode
        if _CUML_OK:
            return cuSVC(kernel="rbf", C=C, gamma=gamma, class_weight=cw)
        return skSVC(kernel="rbf", C=C, gamma=gamma, class_weight=cw, probability=True)
    if clf_name == "SVM-polinomial":
        C = float(params.get("svm_poly_C", 1.0))
        deg = int(params.get("svm_poly_degree", 3))
        coef0 = float(params.get("svm_poly_coef0", 0.0))
        if _CUML_OK:
            return cuSVC(kernel="poly", C=C, degree=deg, coef0=coef0, class_weight=cw)
        return skSVC(kernel="poly", C=C, degree=deg, coef0=coef0,
                     class_weight=cw, probability=True)
    if clf_name == "kNN":
        nn = int(params.get("knn_n_neighbors", 5))
        wts = params.get("knn_weights", "uniform")
        p = int(params.get("knn_p", 2))
        if _CUML_OK:
            return cuKNN(n_neighbors=nn, weights=wts, p=p)
        return skKNN(n_neighbors=nn, weights=wts, p=p)
    if clf_name == "RandomForest":
        n_est = int(params.get("rf_n_estimators", 200))
        max_d = int(params.get("rf_max_depth", 10))
        mss = int(params.get("rf_min_samples_split", 2))
        msl = int(params.get("rf_min_samples_leaf", 1))
        mf = params.get("rf_max_features", "sqrt")
        if _CUML_OK:
            return cuRF(n_estimators=n_est, max_depth=max_d,
                         min_samples_split=mss, min_samples_leaf=msl,
                         random_state=random_state)
        return skRF(n_estimators=n_est, max_depth=max_d,
                     min_samples_split=mss, min_samples_leaf=msl,
                     max_features=mf, class_weight=cw,
                     random_state=random_state, n_jobs=-1)
    raise ValueError(f"Clasificador desconocido: {clf_name}")


# ---------------------------------------------------------------------------
# SMOTE helper
# ---------------------------------------------------------------------------

def _safe_smote_k(y_enc, n_splits):
    min_class_count = int(np.min(np.bincount(y_enc)))
    val_max = -(-min_class_count // n_splits)
    train_min = min_class_count - val_max
    return max(1, min(5, train_min - 1))


# ---------------------------------------------------------------------------
# Métricas binarias (sensibilidad / especificidad / AUC)
# ---------------------------------------------------------------------------

def _binary_metrics(y_true, y_pred, y_prob, pathology_idx: int) -> dict:
    """
    Calcula métricas binarias donde 'pathology' es la clase positiva.
    pathology_idx: índice de 'pathology' en le.classes_ (0 o 1).
    """
    cm = confusion_matrix(y_true, y_pred)
    # cm[i,j]: verdadero=i, predicho=j
    # positivo = pathology (pathology_idx), negativo = healthy (1-pathology_idx)
    pos = pathology_idx
    neg = 1 - pathology_idx
    tp = cm[pos, pos] if cm.shape[0] > pos else 0
    fn = cm[pos, neg] if cm.shape[0] > pos else 0
    fp = cm[neg, pos] if cm.shape[0] > neg else 0
    tn = cm[neg, neg] if cm.shape[0] > neg else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall de pathology
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # recall de healthy
    auc = 0.0
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true == pos, y_prob[:, pos])
        except Exception:
            pass
    return {"sensitivity": sensitivity, "specificity": specificity, "auc": auc}


# ---------------------------------------------------------------------------
# Matriz de confusión y curva ROC (diagnóstico de la evaluación final)
# ---------------------------------------------------------------------------

def _print_confusion_matrix(y_true, y_pred, class_names, title="Matriz de confusión"):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    print(f"\n  {title} (filas=real, columnas=predicho, acumulada sobre todas las semillas):")
    header = "".join(f"{name[:10]:>12}" for name in class_names)
    print(f"  {'':<12}{header}")
    for i, name in enumerate(class_names):
        row_str = "".join(f"{cm[i, j]:>12}" for j in range(len(class_names)))
        print(f"  {name[:12]:<12}{row_str}")


def _plot_roc_curve(y_true, y_prob, class_names, save_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_classes = len(class_names)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fig, ax = plt.subplots(figsize=(6, 6))
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        roc_auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[1]} (AUC={roc_auc:.3f})")
    else:
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i, cname in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Función objetivo — vectorial (B, C, D)
# ---------------------------------------------------------------------------

def _objective(trial, X, y_enc, clf_name, n_splits, balanced, random_state,
               k_best_fixed=None):
    if k_best_fixed is not None:
        k_best = min(k_best_fixed, X.shape[1])
    else:
        k_best = trial.suggest_int("k_best", 20, min(200, X.shape[1]))

    use_smote = trial.suggest_categorical("use_smote", [True, False])
    min_class_count = int(np.min(np.bincount(y_enc)))
    n_splits_eff = max(2, min(n_splits, min_class_count))
    smote_k = _safe_smote_k(y_enc, n_splits_eff)
    can_smote = use_smote and (min_class_count - (-(-min_class_count // n_splits_eff)) >= 2)

    cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    kappas = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_enc)):
        clf = _suggest_and_build_classifier(trial, clf_name, balanced, random_state, fold_idx)
        steps = [
            ("scaler", StandardScaler()),
            ("var", VarianceThreshold(threshold=1e-12)),
            ("select", SelectKBest(score_func=f_classif, k=k_best)),
        ]
        if can_smote:
            steps.append(("smote", SMOTE(k_neighbors=smote_k, random_state=random_state)))
        steps.append(("clf", clf))
        pipe = ImbPipeline(steps)
        try:
            pipe.fit(X[train_idx], y_enc[train_idx])
        except optuna.TrialPruned:
            raise
        except Exception:
            raise optuna.TrialPruned()

        y_pred = pipe.predict(X[val_idx])
        kappas.append(cohen_kappa_score(y_enc[val_idx], y_pred))
        trial.report(float(np.mean(kappas)), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(kappas))


# ---------------------------------------------------------------------------
# Función objetivo — Fisher Vector (E)
# ---------------------------------------------------------------------------

def _objective_fisher(trial, signals, fs, y_enc, clf_name, n_splits, balanced,
                       random_state, n_mels, n_fft, fv_n_components):
    use_smote = trial.suggest_categorical("use_smote", [True, False])
    min_class_count = int(np.min(np.bincount(y_enc)))
    n_splits_eff = max(2, min(n_splits, min_class_count))
    smote_k = _safe_smote_k(y_enc, n_splits_eff)
    can_smote = use_smote and (min_class_count - (-(-min_class_count // n_splits_eff)) >= 2)

    cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    kappas = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(np.zeros(len(y_enc)), y_enc)):
        try:
            train_sfs = [(signals[i], fs) for i in train_idx]
            fv_gmm = fit_fisher_gmm(train_sfs, n_components=fv_n_components,
                                     n_mels=n_mels, n_fft=n_fft, random_state=random_state)
            X_tr = np.array([encode_fisher_vector(signals[i], fs, fv_gmm, n_fft=n_fft)
                              for i in train_idx])
            X_val = np.array([encode_fisher_vector(signals[i], fs, fv_gmm, n_fft=n_fft)
                               for i in val_idx])
        except Exception:
            raise optuna.TrialPruned()

        clf = _suggest_and_build_classifier(trial, clf_name, balanced, random_state, fold_idx)
        steps = [("scaler", StandardScaler()), ("var", VarianceThreshold(threshold=1e-12))]
        if can_smote:
            steps.append(("smote", SMOTE(k_neighbors=smote_k, random_state=random_state)))
        steps.append(("clf", clf))
        pipe = ImbPipeline(steps)
        try:
            pipe.fit(X_tr, y_enc[train_idx])
        except optuna.TrialPruned:
            raise
        except Exception:
            raise optuna.TrialPruned()

        y_pred = pipe.predict(X_val)
        kappas.append(cohen_kappa_score(y_enc[val_idx], y_pred))
        trial.report(float(np.mean(kappas)), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(kappas))


# ---------------------------------------------------------------------------
# Evaluación final multi-semilla — vectorial (B, C, D)
# ---------------------------------------------------------------------------

def _evaluate_best_params(best_params, X, y_enc, clf_name, n_splits, balanced,
                           seeds, le, k_best_fixed=None, pathology_idx=1):
    k_best = min(k_best_fixed if k_best_fixed is not None else int(best_params["k_best"]),
                 X.shape[1])
    use_smote = bool(best_params["use_smote"])
    cw = "balanced" if balanced else None

    results = []
    agg_y_true, agg_y_pred, agg_y_prob = [], [], []
    for seed in seeds:
        min_class_count = int(np.min(np.bincount(y_enc)))
        n_splits_eff = max(2, min(n_splits, min_class_count))
        smote_k = _safe_smote_k(y_enc, n_splits_eff)
        can_smote = use_smote and (
            min_class_count - (-(-min_class_count // n_splits_eff)) >= 2)
        cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)

        y_true_all, y_pred_all, y_prob_all = [], [], []
        for train_idx, val_idx in cv.split(X, y_enc):
            clf = _build_clf_from_params(clf_name, best_params, cw, seed)
            steps = [
                ("scaler", StandardScaler()),
                ("var", VarianceThreshold(threshold=1e-12)),
                ("select", SelectKBest(score_func=f_classif, k=k_best)),
            ]
            if can_smote:
                steps.append(("smote", SMOTE(k_neighbors=smote_k, random_state=seed)))
            steps.append(("clf", clf))
            pipe = ImbPipeline(steps)
            try:
                pipe.fit(X[train_idx], y_enc[train_idx])
                preds = pipe.predict(X[val_idx])
                y_pred_all.extend(preds.tolist())
                y_true_all.extend(y_enc[val_idx].tolist())
                # Probabilidades para AUC (solo si el clf las soporta)
                if hasattr(pipe, "predict_proba"):
                    try:
                        proba = pipe.predict_proba(X[val_idx])
                        y_prob_all.extend(proba.tolist())
                    except Exception:
                        y_prob_all.extend([[0.5, 0.5]] * len(val_idx))
                else:
                    y_prob_all.extend([[0.5, 0.5]] * len(val_idx))
            except Exception:
                pass

        if y_pred_all:
            yt = np.array(y_true_all)
            yp = np.array(y_pred_all)
            yprob = np.array(y_prob_all)
            kappa = cohen_kappa_score(yt, yp)
            acc = accuracy_score(yt, yp)
            rpt = classification_report(yt, yp, target_names=le.classes_,
                                         zero_division=0, output_dict=True)
            bmet = _binary_metrics(yt, yp, yprob, pathology_idx)
            results.append({"kappa": kappa, "acc": acc, "report": rpt, **bmet})
            agg_y_true.extend(y_true_all)
            agg_y_pred.extend(y_pred_all)
            agg_y_prob.extend(y_prob_all)

    agg = _aggregate_results(results, le)
    agg["y_true"] = np.array(agg_y_true)
    agg["y_pred"] = np.array(agg_y_pred)
    agg["y_prob"] = np.array(agg_y_prob) if agg_y_prob else np.zeros((0, len(le.classes_)))
    agg["n_features_used"] = k_best
    return agg


# ---------------------------------------------------------------------------
# Evaluación final multi-semilla — Fisher Vector (E)
# ---------------------------------------------------------------------------

def _evaluate_best_params_fisher(best_params, signals, fs, y_enc, clf_name,
                                  n_splits, balanced, seeds, le,
                                  n_mels, n_fft, fv_n_components, pathology_idx=1):
    use_smote = bool(best_params["use_smote"])
    cw = "balanced" if balanced else None

    results = []
    agg_y_true, agg_y_pred, agg_y_prob = [], [], []
    n_features_used = 2 * n_mels * fv_n_components
    for seed in seeds:
        min_class_count = int(np.min(np.bincount(y_enc)))
        n_splits_eff = max(2, min(n_splits, min_class_count))
        smote_k = _safe_smote_k(y_enc, n_splits_eff)
        can_smote = use_smote and (
            min_class_count - (-(-min_class_count // n_splits_eff)) >= 2)
        cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=seed)

        y_true_all, y_pred_all, y_prob_all = [], [], []
        for train_idx, val_idx in cv.split(np.zeros(len(y_enc)), y_enc):
            try:
                train_sfs = [(signals[i], fs) for i in train_idx]
                fv_gmm = fit_fisher_gmm(train_sfs, n_components=fv_n_components,
                                         n_mels=n_mels, n_fft=n_fft, random_state=seed)
                X_tr = np.array([encode_fisher_vector(signals[i], fs, fv_gmm, n_fft=n_fft)
                                  for i in train_idx])
                X_val = np.array([encode_fisher_vector(signals[i], fs, fv_gmm, n_fft=n_fft)
                                   for i in val_idx])
            except Exception:
                continue

            clf = _build_clf_from_params(clf_name, best_params, cw, seed)
            steps = [("scaler", StandardScaler()), ("var", VarianceThreshold(threshold=1e-12))]
            if can_smote:
                steps.append(("smote", SMOTE(k_neighbors=smote_k, random_state=seed)))
            steps.append(("clf", clf))
            pipe = ImbPipeline(steps)
            try:
                pipe.fit(X_tr, y_enc[train_idx])
                preds = pipe.predict(X_val)
                y_pred_all.extend(preds.tolist())
                y_true_all.extend(y_enc[val_idx].tolist())
                if hasattr(pipe, "predict_proba"):
                    try:
                        proba = pipe.predict_proba(X_val)
                        y_prob_all.extend(proba.tolist())
                    except Exception:
                        y_prob_all.extend([[0.5, 0.5]] * len(val_idx))
                else:
                    y_prob_all.extend([[0.5, 0.5]] * len(val_idx))
            except Exception:
                pass

        if y_pred_all:
            yt = np.array(y_true_all)
            yp = np.array(y_pred_all)
            yprob = np.array(y_prob_all)
            kappa = cohen_kappa_score(yt, yp)
            acc = accuracy_score(yt, yp)
            rpt = classification_report(yt, yp, target_names=le.classes_,
                                         zero_division=0, output_dict=True)
            bmet = _binary_metrics(yt, yp, yprob, pathology_idx)
            results.append({"kappa": kappa, "acc": acc, "report": rpt, **bmet})
            agg_y_true.extend(y_true_all)
            agg_y_pred.extend(y_pred_all)
            agg_y_prob.extend(y_prob_all)

    agg = _aggregate_results(results, le)
    agg["y_true"] = np.array(agg_y_true)
    agg["y_pred"] = np.array(agg_y_pred)
    agg["y_prob"] = np.array(agg_y_prob) if agg_y_prob else np.zeros((0, len(le.classes_)))
    agg["n_features_used"] = n_features_used
    return agg


def _aggregate_results(results, le):
    if not results:
        return {"kappa_mean": 0.0, "kappa_std": 0.0, "acc_mean": 0.0, "acc_std": 0.0,
                "sensitivity_mean": 0.0, "sensitivity_std": 0.0,
                "specificity_mean": 0.0, "specificity_std": 0.0,
                "auc_mean": 0.0, "auc_std": 0.0, "f1_per_class": {}}
    kappas = np.array([r["kappa"] for r in results])
    accs   = np.array([r["acc"]   for r in results])
    sens   = np.array([r["sensitivity"] for r in results])
    spec   = np.array([r["specificity"] for r in results])
    aucs   = np.array([r["auc"] for r in results])
    f1_per_class = {}
    for cls in le.classes_:
        vals = [r["report"].get(cls, {}).get("f1-score", 0.0) for r in results]
        f1_per_class[cls] = (float(np.mean(vals)), float(np.std(vals)))
    return {
        "kappa_mean": float(kappas.mean()), "kappa_std": float(kappas.std()),
        "acc_mean":   float(accs.mean()),   "acc_std":   float(accs.std()),
        "sensitivity_mean": float(sens.mean()), "sensitivity_std": float(sens.std()),
        "specificity_mean": float(spec.mean()), "specificity_std": float(spec.std()),
        "auc_mean":   float(aucs.mean()),   "auc_std":   float(aucs.std()),
        "f1_per_class": f1_per_class,
        "n_seeds": len(results),
    }


# ---------------------------------------------------------------------------
# Guardado
# ---------------------------------------------------------------------------

def _save_trial_history(study, feature_set, clf_name, path):
    rows = []
    for t in study.trials:
        if t.state not in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED):
            continue
        row = {
            "feature_set": feature_set,
            "classifier": clf_name,
            "trial_number": t.number,
            "state": t.state.name,
            "kappa": t.value if t.value is not None else float("nan"),
        }
        row.update(t.params)
        rows.append(row)
    if not rows:
        return
    all_keys = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    write_header = not Path(path).exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _save_summary(summary_rows, path):
    if not summary_rows:
        return
    all_keys = list(summary_rows[0].keys())
    for r in summary_rows[1:]:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)
    print(f"[INFO] Resumen guardado en: {path}")


# ---------------------------------------------------------------------------
# Optimización por conjunto de características
# ---------------------------------------------------------------------------

def run_optuna_binary_feature_set(
    feature_set, X, signals, fs, y_enc, le, clf_names,
    n_trials=100, n_splits=5, balanced=False, n_seeds=5, seed_base=42,
    csv_history="binary_history.csv", n_mels=40, n_fft=1024,
    fv_n_components=16, k_best_fixed=None,
):
    seeds = [seed_base + i for i in range(n_seeds)]
    fs_label = FEATURE_SETS.get(feature_set, feature_set)
    # Índice de 'pathology' en le.classes_
    pathology_idx = list(le.classes_).index("pathology")
    summary_rows = []

    for clf_name in clf_names:
        print(f"\n  {'─'*66}")
        print(f"  [{feature_set}] {fs_label} × {clf_name}  ({n_trials} trials)")
        print(f"  {'─'*66}")
        t0 = time.time()

        pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=3,
                                          min_early_stopping_rate=0)
        sampler = TPESampler(seed=seed_base, multivariate=True, group=True)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner,
            study_name=f"bin_{feature_set}_{clf_name}",
        )

        if feature_set == "E":
            def obj(trial):
                return _objective_fisher(
                    trial, signals, fs, y_enc, clf_name,
                    n_splits=n_splits, balanced=balanced, random_state=seed_base,
                    n_mels=n_mels, n_fft=n_fft, fv_n_components=fv_n_components)
        else:
            def obj(trial):
                return _objective(
                    trial, X, y_enc, clf_name,
                    n_splits=n_splits, balanced=balanced, random_state=seed_base,
                    k_best_fixed=k_best_fixed)

        study.optimize(obj, n_trials=n_trials, show_progress_bar=False)

        elapsed = time.time() - t0
        pruned    = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        completed = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
        print(f"  Completados: {completed}  |  Podados: {pruned}  |  Tiempo: {elapsed:.1f}s")

        if study.best_trial is None or study.best_value is None:
            print(f"  [AVISO] Sin trials completados para {clf_name}.")
            continue

        best_kappa_cv = study.best_value
        best_params = study.best_params
        print(f"  Mejor Kappa (CV): {best_kappa_cv:.4f}")
        for k, v in best_params.items():
            print(f"    {k}: {v}")

        if csv_history:
            _save_trial_history(study, feature_set, clf_name, csv_history)

        print(f"  Evaluando con {n_seeds} semillas...")
        if feature_set == "E":
            final = _evaluate_best_params_fisher(
                best_params, signals, fs, y_enc, clf_name,
                n_splits=n_splits, balanced=balanced, seeds=seeds, le=le,
                n_mels=n_mels, n_fft=n_fft, fv_n_components=fv_n_components,
                pathology_idx=pathology_idx,
            )
        else:
            final = _evaluate_best_params(
                best_params, X, y_enc, clf_name,
                n_splits=n_splits, balanced=balanced, seeds=seeds, le=le,
                k_best_fixed=k_best_fixed, pathology_idx=pathology_idx,
            )

        print(f"  Kappa:       {final['kappa_mean']:.4f} ± {final['kappa_std']:.4f}")
        print(f"  Acc:         {final['acc_mean']:.4f} ± {final['acc_std']:.4f}")
        print(f"  Sensibilidad:{final['sensitivity_mean']:.4f} ± {final['sensitivity_std']:.4f}  "
              f"(recall de pathology)")
        print(f"  Especificidad:{final['specificity_mean']:.4f} ± {final['specificity_std']:.4f}  "
              f"(recall de healthy)")
        print(f"  AUC-ROC:     {final['auc_mean']:.4f} ± {final['auc_std']:.4f}")
        print(f"  Características usadas: {final['n_features_used']}")

        if len(final["y_true"]) > 0:
            _print_confusion_matrix(final["y_true"], final["y_pred"], le.classes_)
            roc_dir = Path("roc_curves_binary")
            roc_dir.mkdir(parents=True, exist_ok=True)
            roc_path = roc_dir / f"roc_{feature_set}_{clf_name}.png"
            _plot_roc_curve(final["y_true"], final["y_prob"], le.classes_, roc_path,
                            title=f"ROC binaria — [{feature_set}] {fs_label} × {clf_name}")
            print(f"  [INFO] Curva ROC guardada en: {roc_path}")

        row = {
            "feature_set": feature_set,
            "feature_set_desc": fs_label,
            "classifier": clf_name,
            "kappa_cv_best": round(best_kappa_cv, 4),
            "kappa_mean": round(final["kappa_mean"], 4),
            "kappa_std": round(final["kappa_std"], 4),
            "acc_mean": round(final["acc_mean"], 4),
            "acc_std": round(final["acc_std"], 4),
            "sensitivity_mean": round(final["sensitivity_mean"], 4),
            "sensitivity_std": round(final["sensitivity_std"], 4),
            "specificity_mean": round(final["specificity_mean"], 4),
            "specificity_std": round(final["specificity_std"], 4),
            "auc_mean": round(final["auc_mean"], 4),
            "auc_std": round(final["auc_std"], 4),
            "n_features_used": final["n_features_used"],
            "n_seeds": n_seeds,
            "n_trials": n_trials,
            "trials_completed": completed,
            "trials_pruned": pruned,
            "elapsed_s": round(elapsed, 1),
        }
        row.update({f"param_{k}": v for k, v in best_params.items()})
        for cls, (f1m, f1s) in final["f1_per_class"].items():
            row[f"f1_{cls}_mean"] = round(f1m, 4)
            row[f"f1_{cls}_std"] = round(f1s, 4)
        summary_rows.append(row)

    return summary_rows


# ---------------------------------------------------------------------------
# Tabla comparativa
# ---------------------------------------------------------------------------

def print_comparison_table(rows):
    if not rows:
        print("[AVISO] Sin resultados para mostrar.")
        return
    rows_sorted = sorted(rows, key=lambda r: r.get("kappa_mean", 0), reverse=True)
    print(f"\n{'='*120}")
    print("CLASIFICACIÓN BINARIA — healthy vs pathology — Comparación por conjunto de características")
    print(f"  healthy   = normal + hunger + pain")
    print(f"  pathology = asphyxia + deaf")
    print(f"{'='*120}")
    hdr = (f"{'Set':<5}{'Descripción':<34}{'Clasificador':<16}"
           f"{'Kappa':>8}{'±':>3}{'Acc':>8}{'Sens':>8}{'Spec':>8}{'AUC':>8}{'OK':>6}{'Pod':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows_sorted:
        desc = r.get("feature_set_desc", "")[:33]
        print(f"{r['feature_set']:<5}{desc:<34}{r['classifier']:<16}"
              f"{r['kappa_mean']:>8.4f}{r['kappa_std']:>3.3f}"
              f"{r['acc_mean']:>8.4f}"
              f"{r['sensitivity_mean']:>8.4f}"
              f"{r['specificity_mean']:>8.4f}"
              f"{r['auc_mean']:>8.4f}"
              f"{r['trials_completed']:>6}{r['trials_pruned']:>6}")
    print()
    best = rows_sorted[0]
    print(f"  ★ Mejor: [{best['feature_set']}] {best.get('feature_set_desc','')} "
          f"× {best['classifier']}")
    print(f"    Kappa {best['kappa_mean']:.4f} ± {best['kappa_std']:.4f}  |  "
          f"Sens {best['sensitivity_mean']:.4f}  |  "
          f"Spec {best['specificity_mean']:.4f}  |  "
          f"AUC {best['auc_mean']:.4f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Clasificación binaria (healthy vs pathology) con Optuna TPE+ASHA"
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--source", default="1s", choices=["full", "1s", "both"])
    parser.add_argument("--duration_s", type=float, default=1.0)
    parser.add_argument("--fs_target", type=int, default=16000)
    parser.add_argument("--no_bandpass", action="store_true")
    parser.add_argument("--low_hz", type=float, default=150.0)
    parser.add_argument("--high_hz", type=float, default=8000.0)
    parser.add_argument("--n_mels", type=int, default=40)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sub_window", type=int, default=4)
    parser.add_argument("--n_mfcc", type=int, default=20)
    parser.add_argument("--f0_fmin", type=float, default=80.0)
    parser.add_argument("--f0_fmax", type=float, default=1000.0)
    parser.add_argument("--k_best_d", type=int, default=60,
                        help="k fijo de SelectKBest para el conjunto D (default: 60)")
    parser.add_argument("--fv_n_components", type=int, default=16,
                        help="Componentes GMM para Fisher Vector (conjunto E)")
    parser.add_argument("--feature_sets", nargs="+", default=["B", "C", "D", "E"],
                        choices=["B", "C", "D", "E"])
    parser.add_argument("--classifiers", nargs="+",
                        default=["MLP", "SVM-lineal", "SVM-RBF", "SVM-polinomial",
                                 "kNN", "RandomForest"])
    parser.add_argument("--n_trials", type=int, default=100)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--skip_analysis", action="store_true")
    parser.add_argument("--csv_history", default="binary_history.csv")
    parser.add_argument("--csv_summary", default="binary_summary.csv")
    args = parser.parse_args()

    report_backend()

    if not args.skip_analysis:
        print(f"\nAnalizando dataset en: {args.data_dir} (source='{args.source}')")
        report = analyze_dataset(args.data_dir, source=args.source)
        print_report(report)

    print(f"\nCargando audios y remapeando a clases binarias...")
    print(f"  healthy   ← {sorted(HEALTHY_CLASSES)}")
    print(f"  pathology ← {sorted(PATHOLOGY_CLASSES)}")
    t_load = time.time()
    signals, fs, y = _load_signals_binary(
        args.data_dir, args.source, args.fs_target, args.duration_s,
        not args.no_bandpass, args.low_hz, args.high_hz,
    )
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"[INFO] {len(signals)} audios listos ({time.time()-t_load:.1f}s)  "
          f"— clases: {list(le.classes_)}\n")

    if args.csv_history and Path(args.csv_history).exists():
        Path(args.csv_history).unlink()

    fs_configs = {
        "B": dict(use_fosp=False, use_complementary=True, use_eefgabor=True),
        "C": dict(use_fosp=True,  use_complementary=True, use_eefgabor=True),
        "D": dict(use_fosp=True,  use_complementary=True, use_eefgabor=True),
        "E": None,
    }

    all_summary_rows = []

    for fs_key in args.feature_sets:
        fs_desc = FEATURE_SETS[fs_key]
        print(f"\n{'='*70}")
        print(f"  CONJUNTO {fs_key}: {fs_desc}")
        print(f"{'='*70}")

        if fs_key == "E":
            X_fs = None
            print("  (Fisher Vector: GMM ajustado dentro de cada fold)")
        else:
            cfg = fs_configs[fs_key]
            print(f"  Extrayendo características "
                  f"(FOSP={'sí' if cfg['use_fosp'] else 'no'}, comp=sí, EEFGabor=sí)...")
            t_ext = time.time()
            X_fs, _ = _extract_vector_features(
                signals, fs, args.n_mels, args.n_fft, args.k, args.sub_window,
                args.n_mfcc, args.f0_fmin, args.f0_fmax, **cfg,
            )
            print(f"  {X_fs.shape[1]} características extraídas ({time.time()-t_ext:.1f}s)")

        k_best_fixed = args.k_best_d if fs_key == "D" else None

        rows = run_optuna_binary_feature_set(
            feature_set=fs_key,
            X=X_fs,
            signals=signals,
            fs=fs,
            y_enc=y_enc,
            le=le,
            clf_names=args.classifiers,
            n_trials=args.n_trials,
            n_splits=args.n_splits,
            balanced=args.balanced,
            n_seeds=args.n_seeds,
            seed_base=args.seed_base,
            csv_history=args.csv_history,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            fv_n_components=args.fv_n_components,
            k_best_fixed=k_best_fixed,
        )
        all_summary_rows.extend(rows)

    print_comparison_table(all_summary_rows)
    _save_summary(all_summary_rows, args.csv_summary)

    if args.csv_history and Path(args.csv_history).exists():
        print(f"[INFO] Histórico de trials en: {args.csv_history}")


if __name__ == "__main__":
    main()
