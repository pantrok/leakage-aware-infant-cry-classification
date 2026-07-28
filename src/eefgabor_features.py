"""
4.3.1 Energía y Entropía de Filtros Gabor (EEFGabor)
========================================================
Adaptación a llanto de bebé del enfoque EEFGabor de la tesis (EEG).

Metodología original (fiel a la tesis):

1) El espectrograma se convierte a una imagen en escala de grises
   (matriz de intensidades/energía).
2) Se aplica un banco de filtros de Gabor 2D, con distintas escalas
   (frecuencias) y orientaciones, vía convolución:
       R(x,y) = A(x,y) ⊛ G(x,y)                      (ec. 4.5)
   donde A es el espectrograma (imagen) y G el filtro de Gabor.
3) De cada respuesta R se extraen dos características de textura:
   - Entropía de Shannon (ec. 4.6): H(X) = -Σ p(xi) log2 p(xi),
     calculada sobre el histograma de niveles de intensidad de R.
   - Energía (ec. 4.7): e(x) = (1/MN) Σ Σ |a(i,j)|, el promedio del
     valor absoluto de la respuesta filtrada.

El banco de filtros (varias escalas x varias orientaciones) genera un
vector de longitud fija: 2 (energía + entropía) x n_escalas x
n_orientaciones.

Adaptación clave: la tesis usa el espectrograma de la señal completa.
Aquí se usa el MISMO Mel-espectrograma que ya calcula spectral_peaks.py
(extract_spectral_peaks_mel) para mantener consistencia con el resto
del pipeline FOSP -- es la imagen de entrada A(x,y) sobre la que se
aplican los filtros Gabor.
"""

from dataclasses import dataclass
import numpy as np
from skimage.filters import gabor_kernel
from scipy.signal import convolve2d

from src.spectral_peaks import compute_mel_spectrogram


@dataclass
class GaborBankConfig:
    frequencies: tuple = (0.05, 0.15, 0.25, 0.35)   # escalas (frecuencias espaciales del filtro)
    orientations: tuple = (0, np.pi/4, np.pi/2, 3*np.pi/4)  # orientaciones (0°, 45°, 90°, 135°)
    sigma: float = 3.0                               # desviación estándar del envolvente gaussiano


def spectrogram_to_grayscale(magnitude: np.ndarray) -> np.ndarray:
    """
    Convierte la magnitud del espectrograma (cualquier escala) a una
    imagen en escala de grises normalizada a [0, 1], análogo a "el
    espectrograma se convierte a imagen" de la tesis (figura 4.4, paso
    3: "Espectrogramas en escala de grises").
    """
    mag = magnitude.astype(float)
    mn, mx = mag.min(), mag.max()
    if mx - mn < 1e-12:
        return np.zeros_like(mag)
    return (mag - mn) / (mx - mn)


def shannon_entropy(image: np.ndarray, n_bins: int = 256) -> float:
    """
    Entropía de Shannon (ec. 4.6) sobre el histograma de niveles de
    intensidad de la imagen (respuesta filtrada), análogo a tratar la
    imagen como "fuente" con distribución de probabilidad p(xi) sobre
    n niveles de intensidad.
    """
    hist, _ = np.histogram(image, bins=n_bins, range=(image.min(), image.max() + 1e-12))
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    p_nonzero = p[p > 0]
    return float(-np.sum(p_nonzero * np.log2(p_nonzero)))


def texture_energy(image: np.ndarray) -> float:
    """
    Energía de textura (ec. 4.7): e(x) = (1/MN) * sum(|a(i,j)|).
    Promedio del valor absoluto de la imagen (respuesta filtrada).
    """
    M, N = image.shape
    return float(np.sum(np.abs(image)) / (M * N))


def build_gabor_bank(config: GaborBankConfig = GaborBankConfig()) -> list[tuple[np.ndarray, float, float]]:
    """
    Construye el banco de kernels de Gabor (uno por cada combinación de
    frecuencia x orientación). Regresa lista de (kernel_real, frequency,
    orientation) para trazabilidad.
    """
    bank = []
    for freq in config.frequencies:
        for theta in config.orientations:
            kernel = gabor_kernel(frequency=freq, theta=theta, sigma_x=config.sigma, sigma_y=config.sigma)
            bank.append((np.real(kernel), freq, theta))
    return bank


def extract_eefgabor_features(magnitude: np.ndarray, config: GaborBankConfig = GaborBankConfig()
                               ) -> tuple[np.ndarray, list[str]]:
    """
    Extrae el vector EEFGabor completo a partir de la magnitud de un
    espectrograma (cualquiera: Mel o lineal).

    Para cada filtro de Gabor en el banco (n_frequencies x n_orientations):
      1. Convoluciona el espectrograma en escala de grises con el filtro
         (ec. 4.5): R(x,y) = A(x,y) ⊛ G(x,y)
      2. Extrae energía (ec. 4.7) y entropía de Shannon (ec. 4.6) de R.

    Regresa (vector, nombres), longitud = 2 * n_frequencies * n_orientations.
    """
    gray = spectrogram_to_grayscale(magnitude)
    bank = build_gabor_bank(config)

    energies = []
    entropies = []
    names = []

    for kernel, freq, theta in bank:
        # mode='same' mantiene el tamaño de la imagen original;
        # boundary='symm' evita artefactos fuertes en los bordes del
        # espectrograma (más estable que 'fill' con ceros para señales
        # cortas como audios de 3s).
        response = convolve2d(gray, kernel, mode="same", boundary="symm")

        e = texture_energy(response)
        h = shannon_entropy(response)

        energies.append(e)
        entropies.append(h)

        theta_deg = int(round(np.degrees(theta)))
        names.append(f"gabor_energy_f{freq:.2f}_t{theta_deg}")
        names.append(f"gabor_entropy_f{freq:.2f}_t{theta_deg}")

    # Intercalar energía/entropía en el mismo orden que los nombres
    vec = []
    for e, h in zip(energies, entropies):
        vec.append(e)
        vec.append(h)

    return np.array(vec), names


def extract_eefgabor_from_audio(signal: np.ndarray, fs: int,
                                 n_mels: int = 40, n_fft: int = 1024,
                                 hop_length: int | None = None,
                                 fmin: float = 0.0, fmax: float | None = None,
                                 config: GaborBankConfig = GaborBankConfig()
                                 ) -> tuple[np.ndarray, list[str]]:
    """
    Conveniencia: calcula el Mel-espectrograma de la señal de audio
    (mismo cálculo que usa FOSP en spectral_peaks.py, para consistencia
    de pipeline) y extrae el vector EEFGabor sobre él.
    """
    fmax_eff = fmax if fmax is not None else fs / 2.0
    _, _, magnitude = compute_mel_spectrogram(
        signal, fs, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax_eff,
    )
    return extract_eefgabor_features(magnitude, config=config)
