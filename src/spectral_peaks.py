"""
4.3.2.1 Picos espectrales
--------------------------
Calcula el espectrograma de una señal de audio (STFT lineal o
Mel-espectrograma) y extrae, por cada ventana de tiempo (columna), el
pico de energía máxima a través de todas las frecuencias:

    E[l] = max[X(l)]

donde X es la magnitud del espectrograma (matriz k x l: k frecuencias o
bandas Mel, l instantes de tiempo).

Para cada ventana l obtenemos:
    - peak_freq_idx[l]  -> índice de frecuencia/banda (fila) del máximo
    - peak_freq_hz[l]   -> frecuencia en Hz correspondiente (centro de
                           banda Mel si domain="mel", o frecuencia lineal
                           si domain="stft")
    - peak_energy[l]    -> valor de energía (magnitud) en ese pico
"""

from dataclasses import dataclass
import numpy as np
from scipy.signal import stft
import librosa


@dataclass
class SpectralPeaks:
    freqs: np.ndarray          # (k,) frecuencias en Hz, eje vertical (centros Mel si domain="mel")
    times: np.ndarray          # (l,) instantes de tiempo en s, eje horizontal
    magnitude: np.ndarray      # (k, l) magnitud del espectrograma
    peak_freq_idx: np.ndarray  # (l,) índice de fila/banda del pico por ventana
    peak_freq_hz: np.ndarray   # (l,) frecuencia del pico por ventana
    peak_energy: np.ndarray    # (l,) energía (magnitud) del pico por ventana
    domain: str = "stft"       # "stft" (lineal) o "mel"


def compute_spectrogram(signal: np.ndarray, fs: int,
                         nperseg: int = 256, noverlap: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la magnitud del espectrograma vía STFT (equivalente a la
    ecuación 4.4 de la tesis: respuesta de la STFT como matriz X con
    k filas (frecuencias) y l columnas (tiempo)).
    """
    if noverlap is None:
        noverlap = nperseg // 2
    freqs, times, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)  # k x l
    return freqs, times, magnitude


def extract_spectral_peaks(signal: np.ndarray, fs: int,
                            nperseg: int = 256, noverlap: int | None = None) -> SpectralPeaks:
    """
    Implementa E[l] = max[X(l)] (ec. 4.8): para cada columna (ventana de
    tiempo) del espectrograma STFT lineal, encuentra el pico de energía
    máxima sobre todas las frecuencias.
    """
    freqs, times, magnitude = compute_spectrogram(signal, fs, nperseg, noverlap)

    peak_freq_idx = np.argmax(magnitude, axis=0)        # (l,)
    peak_energy = magnitude[peak_freq_idx, np.arange(magnitude.shape[1])]
    peak_freq_hz = freqs[peak_freq_idx]

    return SpectralPeaks(
        freqs=freqs,
        times=times,
        magnitude=magnitude,
        peak_freq_idx=peak_freq_idx,
        peak_freq_hz=peak_freq_hz,
        peak_energy=peak_energy,
        domain="stft",
    )


def compute_mel_spectrogram(signal: np.ndarray, fs: int,
                             n_mels: int = 40, n_fft: int = 1024,
                             hop_length: int | None = None,
                             fmin: float = 0.0, fmax: float | None = None
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcula la magnitud del Mel-espectrograma (en dB) de la señal.

    n_mels controla la resolución en el eje de frecuencia perceptual
    (escala Mel, más fina en bajas frecuencias -> más adecuada para voz
    /llanto que la escala lineal de la STFT).

    Regresa:
        mel_freqs : (n_mels,) frecuencia central (Hz) de cada banda Mel
        times     : (l,) instantes de tiempo en segundos
        magnitude : (n_mels, l) energía en dB (log-mel), no negativa tras
                    desplazamiento para mantener compatibilidad con el
                    resto del pipeline (que asume magnitudes >= 0)
    """
    if hop_length is None:
        hop_length = n_fft // 4
    if fmax is None:
        fmax = fs / 2.0

    mel_spec = librosa.feature.melspectrogram(
        y=signal, sr=fs, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0,
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)  # típicamente en [-80, 0] dB
    magnitude = mel_db - mel_db.min()  # desplazar a >= 0 para mantener semántica de "energía"

    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax)
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=fs, hop_length=hop_length)

    return mel_freqs, times, magnitude


def extract_spectral_peaks_mel(signal: np.ndarray, fs: int,
                                n_mels: int = 40, n_fft: int = 1024,
                                hop_length: int | None = None,
                                fmin: float = 0.0, fmax: float | None = None) -> SpectralPeaks:
    """
    Versión del enfoque FOSP (ec. 4.8: E[l] = max[X(l)]) aplicada sobre
    el Mel-espectrograma en vez de la STFT lineal. Recomendado para voz
    /llanto, ya que la escala Mel concentra más resolución en el rango
    de frecuencias perceptualmente relevante (donde se concentra la f0
    del llanto y sus primeros armónicos).
    """
    mel_freqs, times, magnitude = compute_mel_spectrogram(
        signal, fs, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, fmin=fmin, fmax=fmax,
    )

    peak_freq_idx = np.argmax(magnitude, axis=0)
    peak_energy = magnitude[peak_freq_idx, np.arange(magnitude.shape[1])]
    peak_freq_hz = mel_freqs[peak_freq_idx]

    return SpectralPeaks(
        freqs=mel_freqs,
        times=times,
        magnitude=magnitude,
        peak_freq_idx=peak_freq_idx,
        peak_freq_hz=peak_freq_hz,
        peak_energy=peak_energy,
        domain="mel",
    )
