"""Cuenta grabaciones distintas (M) y su desglose por clase.
Uso: python scripts/count_recordings.py --data_dir "ruta/dataset"
"""
import argparse
import re
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.audio_preprocessing import collect_samples


def _recording_id(filepath: str) -> str:
    stem = Path(filepath).stem
    cleaned = re.sub(r'[_\-]\d+$', '', stem)
    return cleaned if cleaned != stem else stem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--source", default="1s", choices=["full", "1s", "both"])
    args = parser.parse_args()

    samples = collect_samples(args.data_dir, source=args.source)
    rec_ids_by_class = {}
    for path, clase in samples:
        rec_ids_by_class.setdefault(clase.lower(), set()).add(_recording_id(path))

    total_recordings = set()
    print(f"{'Clase':<12}{'Grabaciones distintas':>24}{'Segmentos':>14}")
    counts_segments = Counter(c.lower() for _, c in samples)
    for clase, recs in sorted(rec_ids_by_class.items()):
        total_recordings |= recs
        print(f"{clase:<12}{len(recs):>24}{counts_segments[clase]:>14}")
    print(f"\nM (grabaciones distintas, total) = {len(total_recordings)}")
    print(f"N (segmentos de 1s, total)        = {len(samples)}")


if __name__ == "__main__":
    main()
