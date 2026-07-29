"""
Codificación Fisher Vector (FV) sobre Mel-espectrograma
============================================================
Fisher Vector es una técnica de codificación de descriptores locales en
un solo vector de longitud fija por muestra, ampliamente usada en
reconocimiento de imagen/audio antes de la era de deep learning (y aún
competitiva con pocos datos).

Idea: en vez de resumir el espectrograma con estadísticos simples
(media/std, como hace complementary_features.py), se ajusta un GMM
(Gaussian Mixture Model) sobre TODOS los frames (columnas) de TODOS los
espectrogramas del dataset de entrenamiento -- el "vocabulario" de
formas espectrales típicas. Luego, cada audio se codifica calculando el
gradiente del log-likelihood de sus frames respecto a los parámetros del
GMM (medias y varianzas de cada componente), normalizado. Esto captura
no solo "cuánta energía hay en cada banda" (como las features simples)
sino "cómo se desvían los frames del audio respecto al vocabulario
típico", una representación más rica de la distribución completa de
formas espectrales en el audio.

Pasos:
  1. Frame-level: cada columna de tiempo del Mel-espectrograma (vector
     de n_mels valores) es un "descriptor local", análogo a un parche de
     imagen en visión por computadora.
  2. Se ajusta UN SOLO GMM global con todos los frames de TODO el
     conjunto de entrenamiento (no un GMM por audio -- eso no tendría
     sentido, el vocabulario debe ser compartido).
  3. Para cada audio, se calculan los gradientes de Fisher respecto a
     media y varianza de cada componente del GMM, ponderados por la
     responsabilidad posterior de cada frame en cada componente.
  4. Power-normalization (signo * sqrt(|x|)) + L2-normalization, pasos
     estándar en la literatura de Fisher Vectors que mejoran su uso en
     clasificadores lineales/SVM.

CRÍTICO para evitar fuga de datos: el GMM debe ajustarse SOLO con datos
de entrenamiento. fit_fisher_gmm() ajusta el GMM; encode_fisher_vector()
codifica un audio dado un GMM ya ajustado. Los pipelines de este repo
(optuna_search.py, optuna_binary.py, leakage_comparison.py) ajustan el
GMM dentro de cada fold de CV.
"""

from dataclasses import dataclass
import numpy as np
from sklearn.mixture import GaussianMixture

from src.spectral_peaks import compute_mel_spectrogram


@dataclass
class FisherVectorGMM:
    """Envuelve un GMM ajustado junto con metadatos necesarios para codificar."""
    gmm: GaussianMixture
    n_components: int
    n_mels: int


def collect_frames(signals_fs: list[tuple[np.ndarray, int]],
                    n_mels: int = 40, n_fft: int = 1024, hop_length: int | None = None,
                    fmin: float = 0.0, fmax: float | None = None,
                    max_frames_total: int = 50000, random_state: int = 42) -> np.ndarray:
    """
    Calcula el Mel-espectrograma de cada (signal, fs) en la lista y
    concatena todos sus frames (columnas) en una sola matriz
    (n_frames_total, n_mels), para ajustar el GMM global.

    max_frames_total: si el total de frames excede este valor, se
    sub-muestrea aleatoriamente (ajustar un GMM con cientos de miles de
    frames es innecesariamente lento; unos 20-50k es más que suficiente
    para 16 componentes).
    """
    all_frames = []
    for signal, fs in signals_fs:
        fmax_eff = fmax if fmax is not None else fs / 2.0
        _, _, magnitude = compute_mel_spectrogram(
            signal, fs, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax_eff,
        )
        # magnitude: (n_mels, n_frames) -> transponer a (n_frames, n_mels)
        all_frames.append(magnitude.T)

    frames = np.concatenate(all_frames, axis=0)

    if frames.shape[0] > max_frames_total:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(frames.shape[0], size=max_frames_total, replace=False)
        frames = frames[idx]

    return frames


def fit_fisher_gmm(signals_fs: list[tuple[np.ndarray, int]], n_components: int = 16,
                    n_mels: int = 40, n_fft: int = 1024, hop_length: int | None = None,
                    fmin: float = 0.0, fmax: float | None = None,
                    random_state: int = 42) -> FisherVectorGMM:
    """
    Ajusta el GMM global (el "vocabulario" de Fisher Vector) usando los
    frames de TODOS los audios en signals_fs. Debe llamarse SOLO con
    audios de entrenamiento para evitar fuga de datos en validación
    cruzada.
    """
    frames = collect_frames(signals_fs, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
                             fmin=fmin, fmax=fmax, random_state=random_state)

    if frames.shape[0] < 2:
        raise ValueError("No hay suficientes frames para ajustar Fisher Vector")

    # Normalización por frame: cada banda Mel se estandariza usando
    # estadísticos globales de los frames de entrenamiento. Se guardan
    # junto al GMM para aplicar la MISMA normalización al codificar
    # audios nuevos.
    frame_mean = frames.mean(axis=0)
    frame_std = frames.std(axis=0) + 1e-8
    frames_norm = (frames - frame_mean) / frame_std
    frames_norm = frames_norm.astype(np.float64, copy=False)

    n_components_eff = min(n_components, max(1, frames_norm.shape[0] // 10))
    if n_components_eff < n_components:
        print(f"  [INFO] Fisher Vector: reduciendo componentes GMM de {n_components} a {n_components_eff}")

    gmm = GaussianMixture(
        n_components=n_components_eff,
        covariance_type="diag",
        random_state=random_state,
        reg_covar=1e-3,
        max_iter=200,
        n_init=3,
    )
    gmm.fit(frames_norm)

    fv_gmm = FisherVectorGMM(gmm=gmm, n_components=n_components_eff, n_mels=n_mels)
    fv_gmm._frame_mean = frame_mean  # type: ignore[attr-defined]
    fv_gmm._frame_std = frame_std    # type: ignore[attr-defined]
    return fv_gmm


def _fisher_vector_from_frames(frames: np.ndarray, fv_gmm: FisherVectorGMM) -> np.ndarray:
    """
    Calcula el Fisher Vector (gradientes respecto a media y varianza de
    cada componente del GMM) para una matriz de frames (n_frames, n_mels)
    de UN SOLO audio, dado un GMM ya ajustado.

    Implementación estándar (covarianza diagonal):
      Para cada componente k, cada dimensión d:
        gradiente_media[k,d]    = (1/(N*sqrt(w_k))) * sum_t( gamma_t,k * (x_t,d - mu_k,d) / sigma_k,d )
        gradiente_varianza[k,d] = (1/(N*sqrt(2*w_k))) * sum_t( gamma_t,k * ((x_t,d - mu_k,d)^2/sigma_k,d^2 - 1) )
      donde gamma_t,k es la responsabilidad posterior del frame t en el
      componente k (sklearn: gmm.predict_proba).

    Luego: power-normalization (sign(x)*sqrt(|x|)) + L2-normalization.
    """
    frame_mean = fv_gmm._frame_mean  # type: ignore[attr-defined]
    frame_std = fv_gmm._frame_std    # type: ignore[attr-defined]
    frames_norm = (frames - frame_mean) / frame_std
    frames_norm = frames_norm.astype(np.float64, copy=False)

    gmm = fv_gmm.gmm
    N, D = frames_norm.shape
    K = fv_gmm.n_components

    gamma = gmm.predict_proba(frames_norm)  # (N, K)
    weights = gmm.weights_                  # (K,)
    means = gmm.means_                      # (K, D)
    variances = gmm.covariances_            # (K, D) -- covariance_type='diag'

    sqrt_w = np.sqrt(np.maximum(weights, 1e-12))  # (K,)

    diff = frames_norm[:, None, :] - means[None, :, :]       # (N, K, D)
    diff_over_var = diff / variances[None, :, :]             # (N, K, D)

    weighted_diff = gamma[:, :, None] * diff_over_var         # (N, K, D)
    grad_mean = weighted_diff.sum(axis=0)                      # (K, D)
    grad_mean = grad_mean / (N * sqrt_w[:, None])

    term = (diff ** 2) / (variances[None, :, :] ** 2) - (1.0 / variances[None, :, :])
    weighted_term = gamma[:, :, None] * term                   # (N, K, D)
    grad_var = weighted_term.sum(axis=0)                        # (K, D)
    grad_var = grad_var / (N * np.sqrt(2.0) * sqrt_w[:, None])

    fv = np.concatenate([grad_mean.ravel(), grad_var.ravel()])  # (2*K*D,)

    # Power-normalization
    fv = np.sign(fv) * np.sqrt(np.abs(fv) + 1e-12)
    # L2-normalization
    norm = np.linalg.norm(fv)
    if norm > 1e-12:
        fv = fv / norm

    return fv


def encode_fisher_vector(signal: np.ndarray, fs: int, fv_gmm: FisherVectorGMM,
                          n_fft: int = 1024, hop_length: int | None = None,
                          fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """
    Codifica un solo audio como Fisher Vector, dado un GMM ya ajustado
    (típicamente con fit_fisher_gmm sobre el conjunto de entrenamiento).
    """
    fmax_eff = fmax if fmax is not None else fs / 2.0
    _, _, magnitude = compute_mel_spectrogram(
        signal, fs, n_mels=fv_gmm.n_mels, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax_eff,
    )
    frames = magnitude.T  # (n_frames, n_mels)

    if frames.shape[0] == 0:
        return np.zeros(2 * fv_gmm.n_components * fv_gmm.n_mels, dtype=np.float32)

    return _fisher_vector_from_frames(frames, fv_gmm)


def fisher_vector_feature_names(n_components: int, n_mels: int) -> list[str]:
    names = []
    for k in range(n_components):
        for d in range(n_mels):
            names.append(f"fv_mean_k{k}_d{d}")
    for k in range(n_components):
        for d in range(n_mels):
            names.append(f"fv_var_k{k}_d{d}")
    return names
