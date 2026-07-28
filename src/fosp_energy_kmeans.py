"""
4.3.2.3 FOSP con energía, K-means y patrón ternario local
-------------------------------------------------------------
Adaptación a llanto de bebé (audio), reemplazando las "5 bandas conocidas
del EEG (alpha, beta, gamma, delta, theta)" por 5 sub-bandas de frecuencia
del espectro de audio relevantes para llanto infantil (la f0 del llanto
suele ubicarse entre 250-600 Hz, con armónicos hasta varios kHz).

Pasos (fieles a la tesis, sección 4.3.2.3):

1) Energía vía K-means:
   - Se agrupan los picos espectrales (frecuencia, tiempo, energía) con
     K-means, usando distintos valores de K experimentalmente.
   - La característica extraída es la energía en los centroides de cada
     cluster. K es el número de características extraídas (puntos
     máximos más relevantes).

2) Características de tiempo + bandas (celdas):
   - Se forman "secciones o celdas" cruzando ventanas de tiempo x 5
     sub-bandas de frecuencia, donde se ubican los picos máximos.

3) LTP (Patrón Ternario Local):
   - Se compara, para cada ventana n, la celda de banda donde cae el pico
     de esa ventana contra la celda de banda de la ventana n+1:
        0  si están en la misma banda (mismo rango de frecuencia)
        1  si la banda de n es inferior a la de n+1 (el pico "sube")
       -1  si la banda de n es superior a la de n+1 (el pico "baja")
   - El vector ternario (-1, 0, 1) se separa en dos vectores binarios:
       vector_pos: -1 -> 0,  0 -> 0,  1 -> 1   (solo marca subidas)
       vector_neg: -1 -> 1,  0 -> 0,  1 -> 0   (solo marca bajadas)
   - Ambos vectores binarios se convierten a un valor decimal (o, para
     series largas, a un histograma de sub-códigos de longitud fija, que
     es lo que se usa aquí para tener un vector de características de
     tamaño constante entre audios de duración distinta).
"""

from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans

from src.spectral_peaks import SpectralPeaks


# Bandas de frecuencia adaptadas para llanto de bebé.
# La tesis usa 5 bandas EEG (alpha/beta/gamma/delta/theta) de ancho fijo
# en Hz. Aquí, al trabajar sobre Mel-espectrograma, las 5 sub-bandas se
# definen como 5 tramos IGUALES en escala Mel (no en Hz lineal), para
# que cada banda concentre energía perceptualmente comparable -- más
# resolución en frecuencias bajas (donde está la f0 del llanto, ~250-600
# Hz) y menos en las altas, igual que el oído humano.
#
# DEFAULT_CRY_BANDS se mantiene como fallback en Hz lineal por
# compatibilidad hacia atrás (ej. si se usa con extract_spectral_peaks
# en dominio "stft"). Para Mel, usar build_mel_bands().
DEFAULT_CRY_BANDS = [
    (0, 250),       # banda 1: sub-fundamental / ruido de baja frecuencia
    (250, 600),     # banda 2: f0 típica del llanto infantil
    (600, 1200),    # banda 3: primeros armónicos
    (1200, 2500),   # banda 4: armónicos medios / formantes
    (2500, 8000),   # banda 5: armónicos altos / componentes de estrés
]


def hz_to_mel(f_hz: float) -> float:
    return 2595.0 * np.log10(1.0 + f_hz / 700.0)


def mel_to_hz(f_mel: float) -> float:
    return 700.0 * (10.0 ** (f_mel / 2595.0) - 1.0)


def build_mel_bands(fmin: float = 0.0, fmax: float = 8000.0, n_bands: int = 5) -> list[tuple[float, float]]:
    """
    Construye n_bands sub-bandas de ancho igual EN ESCALA MEL entre fmin
    y fmax, y regresa sus límites convertidos de nuevo a Hz. Esto
    reemplaza a las "5 bandas conocidas del EEG" de la tesis, adaptadas
    al dominio Mel del audio de llanto.
    """
    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_edges = np.linspace(mel_min, mel_max, n_bands + 1)
    hz_edges = [mel_to_hz(m) for m in mel_edges]
    return [(hz_edges[i], hz_edges[i + 1]) for i in range(n_bands)]


@dataclass
class FOSPEnergyKmeansFeatures:
    centroid_energy: np.ndarray   # energía en cada centroide (longitud K)
    centroid_freq: np.ndarray     # frecuencia de cada centroide (longitud K)
    centroid_time: np.ndarray     # tiempo de cada centroide (longitud K)
    ltp_pos_hist: np.ndarray      # histograma de sub-códigos "subida"
    ltp_neg_hist: np.ndarray      # histograma de sub-códigos "bajada"
    band_occupancy: np.ndarray    # proporción de picos por banda (longitud 5)


def band_index(freq_hz: float, bands: list[tuple[float, float]] = DEFAULT_CRY_BANDS) -> int:
    for i, (lo, hi) in enumerate(bands):
        if lo <= freq_hz < hi:
            return i
    return len(bands) - 1  # clip al último


def kmeans_energy_features(peaks: SpectralPeaks, k: int = 5, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Agrupa los picos (tiempo, frecuencia, energía) con K-means y extrae,
    por cada centroide: energía, frecuencia y tiempo representativos
    (se usa el centroide directamente, ya en el espacio de las 3
    variables, normalizado para que K-means no esté dominado por la
    escala de Hz vs segundos vs magnitud).
    """
    t = peaks.times
    f = peaks.peak_freq_hz
    e = peaks.peak_energy

    def norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)

    X = np.column_stack([norm(t), norm(f), norm(e)])
    k_eff = min(k, len(X))
    if k_eff < 1:
        return np.zeros(k), np.zeros(k), np.zeros(k)

    km = KMeans(n_clusters=k_eff, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    # Para cada cluster, la "energía en el centroide" se toma como la
    # energía real (no normalizada) del punto más cercano al centroide
    # (más fiel a "energía en los centroides" que un promedio sintético).
    centroid_energy = np.zeros(k_eff)
    centroid_freq = np.zeros(k_eff)
    centroid_time = np.zeros(k_eff)
    for c in range(k_eff):
        idx_in_cluster = np.where(labels == c)[0]
        if len(idx_in_cluster) == 0:
            continue
        cluster_points = X[idx_in_cluster]
        dists = np.linalg.norm(cluster_points - km.cluster_centers_[c], axis=1)
        closest = idx_in_cluster[np.argmin(dists)]
        centroid_energy[c] = e[closest]
        centroid_freq[c] = f[closest]
        centroid_time[c] = t[closest]

    # Si k_eff < k (audio muy corto), se rellena con ceros para vector fijo.
    if k_eff < k:
        pad = k - k_eff
        centroid_energy = np.pad(centroid_energy, (0, pad))
        centroid_freq = np.pad(centroid_freq, (0, pad))
        centroid_time = np.pad(centroid_time, (0, pad))

    return centroid_energy, centroid_freq, centroid_time


def local_ternary_pattern_bands(peak_freq_hz: np.ndarray,
                                 bands: list[tuple[float, float]] = DEFAULT_CRY_BANDS,
                                 sub_window: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """
    LTP adaptado a bandas de frecuencia (en vez de EEG sub-bandas):

    Para cada ventana n, se obtiene la banda (celda) en la que cae su pico
    y se compara con la banda de la ventana n+1:
        0  -> misma banda
        1  -> banda de n+1 es mayor (el pico sube en frecuencia)
       -1  -> banda de n+1 es menor (el pico baja en frecuencia)

    El vector ternario se separa en vector_pos (solo 1s) y vector_neg
    (solo -1s, recodificado a 1s), igual que en la tesis. Cada vector se
    trocea en sub-ventanas de tamaño 'sub_window' y se convierte a un
    código decimal por sub-ventana; el histograma de esos códigos (de
    longitud fija 2^sub_window) es el descriptor final, independiente de
    la duración del audio.
    """
    band_seq = np.array([band_index(f, bands) for f in peak_freq_hz])
    n = len(band_seq)

    ternary = np.zeros(n - 1, dtype=int)
    for i in range(n - 1):
        if band_seq[i + 1] == band_seq[i]:
            ternary[i] = 0
        elif band_seq[i + 1] > band_seq[i]:
            ternary[i] = 1
        else:
            ternary[i] = -1

    vector_pos = (ternary == 1).astype(int)
    vector_neg = (ternary == -1).astype(int)

    def codes_from_binary(vec: np.ndarray, w: int) -> np.ndarray:
        n_full = (len(vec) // w) * w
        if n_full == 0:
            return np.array([], dtype=int)
        trimmed = vec[:n_full].reshape(-1, w)
        weights = 2 ** np.arange(w - 1, -1, -1)
        return trimmed @ weights

    pos_codes = codes_from_binary(vector_pos, sub_window)
    neg_codes = codes_from_binary(vector_neg, sub_window)

    def hist(codes: np.ndarray, w: int) -> np.ndarray:
        n_bins = 2 ** w
        h = np.zeros(n_bins)
        if len(codes) == 0:
            return h
        for c in codes:
            h[c] += 1
        return h / h.sum()

    return hist(pos_codes, sub_window), hist(neg_codes, sub_window)


def band_occupancy(peak_freq_hz: np.ndarray, bands: list[tuple[float, float]] = DEFAULT_CRY_BANDS) -> np.ndarray:
    """Proporción de picos que caen en cada banda (vector de longitud 5)."""
    counts = np.zeros(len(bands))
    for f in peak_freq_hz:
        counts[band_index(f, bands)] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def extract_fosp_energy_kmeans_features(peaks: SpectralPeaks, k: int = 5,
                                         sub_window: int = 4,
                                         bands: list[tuple[float, float]] = DEFAULT_CRY_BANDS
                                         ) -> FOSPEnergyKmeansFeatures:
    centroid_energy, centroid_freq, centroid_time = kmeans_energy_features(peaks, k=k)
    ltp_pos_hist, ltp_neg_hist = local_ternary_pattern_bands(peaks.peak_freq_hz, bands=bands, sub_window=sub_window)
    occupancy = band_occupancy(peaks.peak_freq_hz, bands=bands)

    return FOSPEnergyKmeansFeatures(
        centroid_energy=centroid_energy,
        centroid_freq=centroid_freq,
        centroid_time=centroid_time,
        ltp_pos_hist=ltp_pos_hist,
        ltp_neg_hist=ltp_neg_hist,
        band_occupancy=occupancy,
    )


def fosp_energy_kmeans_to_vector(feat: FOSPEnergyKmeansFeatures) -> np.ndarray:
    """Concatena todas las características en un vector de longitud fija."""
    return np.concatenate([
        feat.centroid_energy,
        feat.centroid_freq,
        feat.band_occupancy,
        feat.ltp_pos_hist,
        feat.ltp_neg_hist,
    ])
