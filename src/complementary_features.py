"""
Características complementarias a FOSP: MFCC, prosódicas (F0) y
espectrales clásicas
========================================================================
FOSP (spectral_peaks.py + fosp_surface.py + fosp_energy_kmeans.py) está
orientado a picos discretos de energía/frecuencia y resúmenes globales
(volumen, área, energía total) — captura bien "dónde y cuán intensos son
los eventos espectrales", pero pierde dos cosas importantes para
distinguir TIPOS de llanto:

1) Timbre fino: la forma detallada del envolvente espectral (MFCC),
   que es justo lo que distingue calidad de voz/llanto (tenso, ronco,
   nasal, etc.) y es el estándar en reconocimiento de voz/llanto.

2) Melodía/prosodia: el contorno de F0 a lo largo del tiempo (sube,
   baja, se mantiene plano, vibra) — un llanto de dolor típicamente
   tiene F0 alta y muy variable; uno de hambre suele ser más rítmico
   y de F0 más estable; asphyxia/deaf suelen tener patrones de F0 y
   energía atípicos (más planos, sin la melodía típica del llanto).

Todas las funciones aquí devuelven vectores de características de
LONGITUD FIJA (estadísticos resumen: media, std, min, max, etc.) para
que se puedan concatenar directamente con el vector FOSP sin importar
la duración del audio.
"""

from dataclasses import dataclass, field
import numpy as np
import librosa


# ---------------------------------------------------------------------
# 1) MFCC + delta + delta-delta
# ---------------------------------------------------------------------

def extract_mfcc_features(signal: np.ndarray, fs: int, n_mfcc: int = 20,
                           n_fft: int = 1024, hop_length: int | None = None) -> np.ndarray:
    """
    Extrae MFCC y sus derivadas (delta, delta-delta), resumidas con
    media y desviación estándar a lo largo del tiempo -> vector de
    longitud 6 * n_mfcc (mean+std de mfcc, delta, delta2).

    MFCC captura el timbre del llanto (forma del envolvente espectral
    en escala perceptual), complementario a los picos discretos de FOSP.
    """
    if hop_length is None:
        hop_length = n_fft // 4

    mfcc = librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    def stats(mat: np.ndarray) -> np.ndarray:
        return np.concatenate([mat.mean(axis=1), mat.std(axis=1)])

    return np.concatenate([stats(mfcc), stats(delta), stats(delta2)])


# ---------------------------------------------------------------------
# 2) Prosódicas: F0 (pitch), jitter, shimmer
# ---------------------------------------------------------------------

@dataclass
class ProsodyFeatures:
    f0_mean: float
    f0_std: float
    f0_min: float
    f0_max: float
    f0_range: float
    voiced_ratio: float     # proporción de tramas con f0 detectado (no silencio/no-armónico)
    jitter: float           # variabilidad ciclo-a-ciclo de f0 (perturbación de frecuencia)
    shimmer: float          # variabilidad ciclo-a-ciclo de amplitud (perturbación de amplitud)


def extract_prosody_features(signal: np.ndarray, fs: int,
                              fmin: float = 80.0, fmax: float = 1000.0,
                              frame_length: int = 1024, hop_length: int | None = None
                              ) -> ProsodyFeatures:
    """
    Extrae el contorno de F0 con el algoritmo pYIN (robusto a ruido,
    da probabilidad de voz por trama) y calcula estadísticos del
    contorno + jitter/shimmer (medidas estándar de calidad de voz,
    usadas en patología de la voz y llanto infantil patológico).

    fmin/fmax=80-1000 Hz cubre la f0 típica del llanto infantil (~250-
    600 Hz) con margen para casos atípicos (asphyxia/deaf con F0 fuera
    de rango normal).
    """
    if hop_length is None:
        hop_length = frame_length // 4

    f0, voiced_flag, voiced_prob = librosa.pyin(
        signal, fmin=fmin, fmax=fmax, sr=fs,
        frame_length=frame_length, hop_length=hop_length,
    )

    voiced_ratio = float(np.mean(voiced_flag)) if len(voiced_flag) > 0 else 0.0
    f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([])
    f0_voiced = f0_voiced[~np.isnan(f0_voiced)]

    if len(f0_voiced) < 2:
        # Sin suficiente voz detectada (señal muy ruidosa, no llanto claro)
        return ProsodyFeatures(
            f0_mean=0.0, f0_std=0.0, f0_min=0.0, f0_max=0.0, f0_range=0.0,
            voiced_ratio=voiced_ratio, jitter=0.0, shimmer=0.0,
        )

    f0_mean = float(np.mean(f0_voiced))
    f0_std = float(np.std(f0_voiced))
    f0_min = float(np.min(f0_voiced))
    f0_max = float(np.max(f0_voiced))
    f0_range = f0_max - f0_min

    # Jitter: variabilidad relativa ciclo-a-ciclo de F0 (absoluta, normalizada por la media)
    f0_diffs = np.abs(np.diff(f0_voiced))
    jitter = float(np.mean(f0_diffs) / f0_mean) if f0_mean > 0 else 0.0

    # Shimmer: variabilidad relativa ciclo-a-ciclo de amplitud (usamos RMS por trama como proxy)
    rms = librosa.feature.rms(y=signal, frame_length=frame_length, hop_length=hop_length)[0]
    rms_voiced = rms[:len(voiced_flag)][voiced_flag] if len(rms) >= len(voiced_flag) else rms
    rms_voiced = rms_voiced[rms_voiced > 0]
    if len(rms_voiced) >= 2:
        rms_diffs = np.abs(np.diff(rms_voiced))
        shimmer = float(np.mean(rms_diffs) / np.mean(rms_voiced)) if np.mean(rms_voiced) > 0 else 0.0
    else:
        shimmer = 0.0

    return ProsodyFeatures(
        f0_mean=f0_mean, f0_std=f0_std, f0_min=f0_min, f0_max=f0_max, f0_range=f0_range,
        voiced_ratio=voiced_ratio, jitter=jitter, shimmer=shimmer,
    )


def prosody_to_vector(feat: ProsodyFeatures) -> np.ndarray:
    return np.array([
        feat.f0_mean, feat.f0_std, feat.f0_min, feat.f0_max, feat.f0_range,
        feat.voiced_ratio, feat.jitter, feat.shimmer,
    ])


# ---------------------------------------------------------------------
# 3) Espectrales clásicas: centroide, ancho de banda, rolloff, ZCR, flatness
# ---------------------------------------------------------------------

def extract_classic_spectral_features(signal: np.ndarray, fs: int,
                                       n_fft: int = 1024, hop_length: int | None = None) -> np.ndarray:
    """
    Extrae descriptores espectrales clásicos de reconocimiento de audio,
    cada uno resumido con media y desviación estándar a lo largo del
    tiempo:
        - Centroide espectral: "brillo" del sonido
        - Ancho de banda espectral: dispersión de energía alrededor del centroide
        - Roll-off espectral (85%): frecuencia bajo la cual cae el 85% de la energía
        - Zero-crossing rate: proxy de ruido/tonalidad (más alto = más ruidoso/fricativo)
        - Flatness espectral: qué tan "plano" (ruidoso) vs "tonal" es el espectro

    Vector resultante: longitud 10 (5 descriptores x mean+std).
    """
    if hop_length is None:
        hop_length = n_fft // 4

    centroid = librosa.feature.spectral_centroid(y=signal, sr=fs, n_fft=n_fft, hop_length=hop_length)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=fs, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=fs, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)[0]
    zcr = librosa.feature.zero_crossing_rate(signal, frame_length=n_fft, hop_length=hop_length)[0]
    flatness = librosa.feature.spectral_flatness(y=signal, n_fft=n_fft, hop_length=hop_length)[0]

    def stats(v: np.ndarray) -> tuple:
        return float(v.mean()), float(v.std())

    out = []
    for v in (centroid, bandwidth, rolloff, zcr, flatness):
        out.extend(stats(v))
    return np.array(out)


# ---------------------------------------------------------------------
# Vector combinado
# ---------------------------------------------------------------------

@dataclass
class ComplementaryFeatureNames:
    """Nombres descriptivos para trazabilidad (usados en selección de features)."""
    names: list = field(default_factory=list)


def extract_complementary_features(signal: np.ndarray, fs: int,
                                    n_mfcc: int = 20, n_fft: int = 1024,
                                    f0_fmin: float = 80.0, f0_fmax: float = 1000.0
                                    ) -> tuple[np.ndarray, list[str]]:
    """
    Extrae y concatena MFCC+deltas, prosódicas y espectrales clásicas en
    un solo vector, junto con una lista de nombres (para trazabilidad
    en la selección de características).
    """
    mfcc_vec = extract_mfcc_features(signal, fs, n_mfcc=n_mfcc, n_fft=n_fft)
    prosody_feat = extract_prosody_features(signal, fs, fmin=f0_fmin, fmax=f0_fmax)
    prosody_vec = prosody_to_vector(prosody_feat)
    spectral_vec = extract_classic_spectral_features(signal, fs, n_fft=n_fft)

    vec = np.concatenate([mfcc_vec, prosody_vec, spectral_vec])

    names = []
    for stage in ["mfcc", "delta", "delta2"]:
        for stat in ["mean", "std"]:
            for i in range(n_mfcc):
                names.append(f"{stage}_{stat}_{i}")
    names += ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_range",
              "voiced_ratio", "jitter", "shimmer"]
    for desc in ["centroid", "bandwidth", "rolloff", "zcr", "flatness"]:
        names += [f"{desc}_mean", f"{desc}_std"]

    return vec, names
