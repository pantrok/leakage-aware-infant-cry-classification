# Results

These files are the outputs cited in the paper — copied here as-is for
traceability, not regenerated. They are evidence, not executable source.

| File | Produced by | Paper reference |
|---|---|---|
| `optuna_summary_multiclass.csv`, `optuna_history_multiclass.csv` | `scripts/optuna_search.py` (`python main.py multiclass`) | Tables 2–3 |
| `binary_summary.csv`, `binary_history.csv` | `scripts/optuna_binary.py` (`python main.py binary`) | Table 4 |
| `leakage_results.csv`, `leakage_stats.csv`, `leakage_plots/` | `scripts/leakage_comparison.py` (`python main.py leakage`) | Section 4.3, Tables 5–6, Figures 6–8 |

Re-running the corresponding command with the same `--data_dir` and default
flags reproduces these files; see the main [README](../README.md) for setup.
