"""
Preprocesamiento de audio para el dataset de llanto de bebé
================================================================
Responsabilidades:

1) Mapeo de carpetas -> clase unificada.
   Estructura real del dataset (tipo Baby Chillanto / similar):
       1s_X       -> versión segmentada a 1s de Full_X (MISMO llanto)
       Full_X      -> grabación completa de la clase X
       FullNormal  -> grabación completa de la clase "normal"
   Por diseño, 1s_X es derivado de Full_X (mismo audio fuente), así que
   usar ambos como muestras "independientes" causaría fuga de datos
   (data leakage) entre entrenamiento y prueba. Por default este pipeline
   usa SOLO las carpetas Full_X / FullNormal como fuente de muestras.

2) Recorte / relleno a duración fija (3 s por defecto), para que todas
   las muestras generen vectores de características de la misma longitud
   y no haya sesgo por duración.

3) Filtro paso-banda para atenuar ruido de fondo de hospital
   (climatizadores, voces lejanas, equipo médico) fuera del rango
   relevante de llanto infantil (~150–8000 Hz aprox.).
"""

import os
import re
import glob
from dataclasses import dataclass

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt


# ---------------------------------------------------------------------
# 1) Mapeo de carpetas a clase unificada
# ---------------------------------------------------------------------

def folder_to_class(folder_name: str) -> tuple[str, str]:
    """
    Dado el nombre de una carpeta, regresa (clase_unificada, tipo_fuente)
    donde tipo_fuente in {"1s", "full"}.

    Ejemplos:
        "1s_asphyxia" -> ("asphyxia", "1s")
        "Full_asphyxia" -> ("asphyxia", "full")
        "FullNormal" -> ("normal", "full")
    """
    name = folder_name.strip()

    # Caso especial: FullNormal (sin guión bajo)
    if name.lower() == "fullnormal":
        return "normal", "full"

    m = re.match(r"^(1s|Full)_(.+)$", name, flags=re.IGNORECASE)
    if m:
        prefix, clase = m.group(1).lower(), m.group(2).lower()
        tipo = "1s" if prefix == "1s" else "full"
        return clase, tipo

    # Si no matchea ningún patrón conocido, se usa el nombre tal cual
    # (en minúsculas) como clase, tipo "full" por default.
    return name.lower(), "full"


def discover_classes(data_dir: str, source: str = "full") -> dict[str, list[str]]:
    """
    Recorre data_dir y agrupa carpetas por clase unificada.

    Parámetros
    ----------
    source: "full"  -> usa solo carpetas Full_X / FullNormal (recomendado,
                        evita fuga de datos por solapamiento con 1s_X).
            "1s"    -> usa solo carpetas 1s_X.
            "both"  -> usa ambas (advertencia: puede causar fuga de datos
                       si luego separas train/test por muestra y no por
                       grabación original).

    Regresa: {clase_unificada: [lista_de_carpetas_absolutas]}
    """
    assert source in ("full", "1s", "both")

    subdirs = sorted([d for d in os.listdir(data_dir)
                       if os.path.isdir(os.path.join(data_dir, d))])

    class_map: dict[str, list[str]] = {}
    for d in subdirs:
        clase, tipo = folder_to_class(d)
        if source != "both" and tipo != source:
            continue
        class_map.setdefault(clase, []).append(os.path.join(data_dir, d))

    return class_map


# ---------------------------------------------------------------------
# 2) Carga + recorte/relleno a duración fija
# ---------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 16000) -> np.ndarray:
    """Carga un audio (cualquier formato soportado por librosa/soundfile),
    lo convierte a mono y resamplea a target_sr."""
    signal, _ = librosa.load(path, sr=target_sr, mono=True)
    return signal.astype(np.float32)


def fix_length(signal: np.ndarray, fs: int, duration_s: float = 3.0) -> np.ndarray:
    """
    Recorta o rellena (zero-pad) la señal a una duración fija en segundos.

    - Si es más larga: se toma el segmento central de duración_s
      (evita silencios de inicio/fin, comunes en grabaciones clínicas
      donde el llanto no llena toda la pista).
    - Si es más corta: se rellena con ceros, centrado.
    """
    target_len = int(duration_s * fs)
    n = len(signal)

    if n == target_len:
        return signal
    if n > target_len:
        start = (n - target_len) // 2
        return signal[start:start + target_len]

    # rellenar con ceros, centrado
    pad_total = target_len - n
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(signal, (pad_left, pad_right), mode="constant")


# ---------------------------------------------------------------------
# 3) Filtro paso-banda (atenuar ruido de hospital)
# ---------------------------------------------------------------------

def bandpass_filter(signal: np.ndarray, fs: int,
                     low_hz: float = 150.0, high_hz: float = 8000.0,
                     order: int = 6) -> np.ndarray:
    """
    Filtro Butterworth paso-banda (forward-backward, sin desfase) que
    atenúa:
      - Ruido de baja frecuencia (< low_hz): zumbido eléctrico, ruido de
        climatización/ventiladores de hospital, manejo de micrófono.
      - Ruido de alta frecuencia (> high_hz): siseo electrónico, parte
        del ruido ambiental de equipo médico.
    El rango [150, 8000] Hz cubre la f0 del llanto infantil (~250-600 Hz)
    y sus armónicos relevantes, dejando fuera buena parte del ruido de
    fondo típico de un entorno hospitalario.
    """
    nyq = fs / 2.0
    high_hz_eff = min(high_hz, nyq * 0.99)  # evitar exceder Nyquist
    sos = butter(order, [low_hz / nyq, high_hz_eff / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


@dataclass
class AudioSample:
    path: str
    class_name: str
    signal: np.ndarray
    fs: int


def load_and_preprocess(path: str, class_name: str, fs_target: int = 16000,
                         duration_s: float = 3.0,
                         apply_bandpass: bool = True,
                         low_hz: float = 150.0, high_hz: float = 8000.0) -> AudioSample:
    """Carga, filtra (paso-banda) y fija duración. Orden: filtrar primero
    sobre la señal completa (mejor estimación espectral), luego recortar."""
    signal = load_audio(path, target_sr=fs_target)
    if apply_bandpass:
        signal = bandpass_filter(signal, fs_target, low_hz=low_hz, high_hz=high_hz)
    signal = fix_length(signal, fs_target, duration_s=duration_s)
    return AudioSample(path=path, class_name=class_name, signal=signal, fs=fs_target)


def collect_samples(data_dir: str, source: str = "full",
                     extensions: tuple[str, ...] = (".wav",)) -> list[tuple[str, str]]:
    """
    Regresa lista de (filepath, clase_unificada) recorriendo las carpetas
    correspondientes según 'source'.
    """
    class_map = discover_classes(data_dir, source=source)
    if not class_map:
        raise ValueError(f"No se encontraron carpetas de clase tipo '{source}' en {data_dir}")

    samples = []
    for clase, folders in class_map.items():
        for folder in folders:
            for ext in extensions:
                for f in glob.glob(os.path.join(folder, f"*{ext}")):
                    samples.append((f, clase))
    return samples
