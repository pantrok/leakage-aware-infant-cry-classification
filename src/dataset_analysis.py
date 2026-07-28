"""
Análisis exploratorio del dataset (desbalance de clases, duraciones)
========================================================================
Genera un reporte (texto + gráfica opcional) ANTES de extraer
características, para diagnosticar:
  - Desbalance entre clases (conteo, proporciones, índice de desbalance)
  - Distribución de duraciones de audio por clase (para justificar la
    duración fija elegida en fix_length)
"""

import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from src.audio_preprocessing import collect_samples


@dataclass
class DatasetReport:
    class_counts: dict[str, int]
    total_samples: int
    imbalance_ratio: float          # max_count / min_count
    duration_stats: dict[str, dict] # por clase: {mean, std, min, max, n}


def get_duration(path: str) -> float:
    """Duración en segundos sin cargar toda la señal en memoria innecesariamente."""
    info = sf.info(path)
    return info.frames / info.samplerate


def analyze_dataset(data_dir: str, source: str = "full") -> DatasetReport:
    samples = collect_samples(data_dir, source=source)  # [(path, clase), ...]

    class_counts: dict[str, int] = {}
    durations_by_class: dict[str, list[float]] = {}

    for path, clase in samples:
        class_counts[clase] = class_counts.get(clase, 0) + 1
        try:
            dur = get_duration(path)
        except Exception:
            dur = float("nan")
        durations_by_class.setdefault(clase, []).append(dur)

    total = sum(class_counts.values())
    max_c = max(class_counts.values())
    min_c = min(class_counts.values())
    imbalance_ratio = max_c / min_c if min_c > 0 else float("inf")

    duration_stats = {}
    for clase, durs in durations_by_class.items():
        durs_arr = np.array([d for d in durs if not np.isnan(d)])
        duration_stats[clase] = {
            "n": len(durs_arr),
            "mean": float(durs_arr.mean()) if len(durs_arr) else float("nan"),
            "std": float(durs_arr.std()) if len(durs_arr) else float("nan"),
            "min": float(durs_arr.min()) if len(durs_arr) else float("nan"),
            "max": float(durs_arr.max()) if len(durs_arr) else float("nan"),
        }

    return DatasetReport(
        class_counts=class_counts,
        total_samples=total,
        imbalance_ratio=imbalance_ratio,
        duration_stats=duration_stats,
    )


def print_report(report: DatasetReport):
    print("=" * 60)
    print("ANÁLISIS EXPLORATORIO DEL DATASET")
    print("=" * 60)
    print(f"\nTotal de muestras: {report.total_samples}")
    print(f"Número de clases: {len(report.class_counts)}")
    print(f"Razón de desbalance (max/min): {report.imbalance_ratio:.2f}\n")

    print(f"{'Clase':<15}{'N':>6}{'%':>8}{'Dur.media(s)':>15}{'Dur.std':>10}{'Dur.min':>10}{'Dur.max':>10}")
    print("-" * 80)
    for clase in sorted(report.class_counts.keys()):
        n = report.class_counts[clase]
        pct = 100 * n / report.total_samples
        ds = report.duration_stats[clase]
        print(f"{clase:<15}{n:>6}{pct:>7.1f}%{ds['mean']:>14.2f}{ds['std']:>10.2f}{ds['min']:>10.2f}{ds['max']:>10.2f}")

    print("\n" + "=" * 60)
    if report.imbalance_ratio > 3:
        print("AVISO: desbalance considerable (>3x). Se recomienda usar")
        print("class_weight='balanced' en los clasificadores, o aplicar")
        print("sobre/sub-muestreo (ej. SMOTE) antes de entrenar.")
    print("=" * 60)


def plot_report(report: DatasetReport, save_path: str | None = None):
    """Genera gráfica de barras del conteo por clase. Requiere matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = sorted(report.class_counts.keys())
    counts = [report.class_counts[c] for c in classes]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(classes, counts, color="#4C72B0")
    ax.set_ylabel("Número de muestras")
    ax.set_title("Distribución de clases (desbalance)")
    for bar, c in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 str(c), ha="center", va="bottom")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Gráfica guardada en: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--source", default="full", choices=["full", "1s", "both"])
    parser.add_argument("--plot", default=None, help="Ruta para guardar gráfica de barras (png)")
    args = parser.parse_args()

    report = analyze_dataset(args.data_dir, source=args.source)
    print_report(report)
    if args.plot:
        plot_report(report, save_path=args.plot)
