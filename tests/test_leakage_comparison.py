import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.leakage_comparison import _save_csv, main


def test_save_csv_creates_file_even_when_rows_empty(tmp_path: Path) -> None:
    out_path = tmp_path / "empty.csv"

    _save_csv([], str(out_path))

    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == ""


def test_main_writes_csvs_even_if_run_comparison_fails(tmp_path: Path, monkeypatch) -> None:
    out_results = tmp_path / "results.csv"
    out_stats = tmp_path / "stats.csv"

    monkeypatch.setattr(
        "scripts.leakage_comparison.run_comparison",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "leakage_comparison.py",
            "--data_dir",
            "dummy",
            "--skip_analysis",
            "--csv_results",
            str(out_results),
            "--csv_stats",
            str(out_stats),
            "--feature_sets",
            "B",
            "--modes",
            "multiclass",
            "--classifiers",
            "MLP",
        ],
    )

    main()

    assert out_results.exists()
    assert out_stats.exists()
    assert out_results.read_text(encoding="utf-8") == ""
    assert out_stats.read_text(encoding="utf-8") == ""
