"""
4.3.2.2 FOSP con superficies, frecuencia y patrón binario local
------------------------------------------------------------------
Tres grupos de características a partir del espectrograma y sus picos:

1) Volumen y área de superficie (cerco convexo / convex hull):
   - "Todos los puntos de intensidad y las coordenadas de tiempo y
     frecuencia se usaron para obtener el volumen del espectrograma" ->
     se aproxima con el volumen del convex hull en 3D (tiempo, frecuencia,
     energía) sobre todos los puntos del espectrograma.
   - "Los picos espectrales se usaron para calcular el área de superficie"
     -> convex hull en 2D (tiempo, frecuencia) o 3D solo de los picos.
   - Energía aproximada = suma de todas las intensidades del espectrograma.

2) Frecuencia: la frecuencia de cada pico máximo por ventana (ya calculada
   en spectral_peaks.py), resumida con estadísticos.

3) Patrón Binario Local (LBP) temporal adaptado:
   "compara el nivel de frecuencia pico de cada ventana a lo largo del
   tiempo y genera un código" -> LBP 1D sobre la secuencia de índices de
   frecuencia pico: para cada ventana l, se compara peak_freq_idx[l] con
   sus vecinos (l-1, l+1, ...) generando bits 1/0 si el vecino es mayor o
   igual / menor, y se decodifica a un entero. Esto es análogo al LBP
   clásico (comparación de un píxel central con su vecindad) pero
   aplicado a la serie temporal de picos.
"""

from dataclasses import dataclass
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from src.spectral_peaks import SpectralPeaks


@dataclass
class FOSPSurfaceFeatures:
    volume_full: float          # volumen convex hull de todo el espectrograma (t, f, E)
    surface_area_peaks: float   # área convex hull de los picos (t, f) o (t, f, E)
    total_energy: float         # suma de todas las intensidades del espectrograma
    freq_mean: float
    freq_std: float
    freq_min: float
    freq_max: float
    lbp_codes: np.ndarray       # códigos LBP por ventana (vector de enteros)
    lbp_hist: np.ndarray        # histograma normalizado de códigos LBP (vector de longitud fija)


def _safe_convex_hull_metric(points: np.ndarray, metric: str = "volume") -> float:
    """
    Calcula volumen (3D) o área (2D/3D, según 'metric') del cerco convexo.
    Si los puntos son degenerados (coplanares/colineales) o insuficientes,
    regresa 0.0 en vez de fallar.
    """
    try:
        hull = ConvexHull(points)
        return float(hull.volume if metric == "volume" else hull.area)
    except QhullError:
        return 0.0
    except Exception:
        return 0.0


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    rng = x.max() - x.min()
    if rng == 0:
        return np.zeros_like(x)
    return (x - x.min()) / rng


def local_binary_pattern_temporal(peak_freq_idx: np.ndarray, neighborhood: int = 3) -> np.ndarray:
    """
    LBP adaptado a la serie temporal de índices de frecuencia pico.

    Para cada ventana central c, se compara su valor con 'neighborhood'
    vecinos hacia adelante (replicando la idea de "comparar la celda del
    pico en la ventana n con la celda del pico en la ventana n+1" de la
    tesis, generalizado a una vecindad mayor si se desea).

    bit_i = 1 si vecino_i >= centro, 0 si vecino_i < centro

    Los bits se concatenan y convierten a un entero decimal (código LBP).
    Bordes (las últimas 'neighborhood' ventanas) se descartan ya que no
    tienen suficientes vecinos futuros.
    """
    n = len(peak_freq_idx)
    codes = []
    for c in range(n - neighborhood):
        center = peak_freq_idx[c]
        bits = []
        for k in range(1, neighborhood + 1):
            neighbor = peak_freq_idx[c + k]
            bits.append(1 if neighbor >= center else 0)
        code = int("".join(map(str, bits)), 2)
        codes.append(code)
    return np.array(codes, dtype=int)


def lbp_histogram(codes: np.ndarray, neighborhood: int = 3) -> np.ndarray:
    """Histograma normalizado de códigos LBP (2^neighborhood bins)."""
    n_bins = 2 ** neighborhood
    hist = np.zeros(n_bins)
    if len(codes) == 0:
        return hist
    for c in codes:
        hist[c] += 1
    return hist / hist.sum()


def extract_fosp_surface_features(peaks: SpectralPeaks, lbp_neighborhood: int = 3) -> FOSPSurfaceFeatures:
    k, l = peaks.magnitude.shape

    # --- 1) Volumen del espectrograma completo (todos los puntos: t, f, E) ---
    time_grid, freq_grid = np.meshgrid(peaks.times, peaks.freqs)  # (k, l) cada uno
    all_points = np.column_stack([
        _normalize(time_grid.ravel()),
        _normalize(freq_grid.ravel()),
        _normalize(peaks.magnitude.ravel()),
    ])
    # Para evitar costo excesivo en espectrogramas grandes, se puede
    # sub-muestrear; aquí se usa todo si es manejable.
    volume_full = _safe_convex_hull_metric(all_points, metric="volume")

    # --- 2) Área de superficie a partir de los picos (t, f, E) ---
    peak_points = np.column_stack([
        _normalize(peaks.times),
        _normalize(peaks.peak_freq_hz),
        _normalize(peaks.peak_energy),
    ])
    surface_area_peaks = _safe_convex_hull_metric(peak_points, metric="area")

    # --- Energía aproximada total ---
    total_energy = float(peaks.magnitude.sum())

    # --- Frecuencia: estadísticos sobre la frecuencia de los picos ---
    freq_mean = float(peaks.peak_freq_hz.mean())
    freq_std = float(peaks.peak_freq_hz.std())
    freq_min = float(peaks.peak_freq_hz.min())
    freq_max = float(peaks.peak_freq_hz.max())

    # --- 3) LBP temporal sobre índices de frecuencia pico ---
    lbp_codes = local_binary_pattern_temporal(peaks.peak_freq_idx, neighborhood=lbp_neighborhood)
    lbp_hist = lbp_histogram(lbp_codes, neighborhood=lbp_neighborhood)

    return FOSPSurfaceFeatures(
        volume_full=volume_full,
        surface_area_peaks=surface_area_peaks,
        total_energy=total_energy,
        freq_mean=freq_mean,
        freq_std=freq_std,
        freq_min=freq_min,
        freq_max=freq_max,
        lbp_codes=lbp_codes,
        lbp_hist=lbp_hist,
    )


def fosp_surface_to_vector(feat: FOSPSurfaceFeatures) -> np.ndarray:
    """Concatena todas las características numéricas en un solo vector fijo."""
    scalars = np.array([
        feat.volume_full,
        feat.surface_area_peaks,
        feat.total_energy,
        feat.freq_mean,
        feat.freq_std,
        feat.freq_min,
        feat.freq_max,
    ])
    return np.concatenate([scalars, feat.lbp_hist])
