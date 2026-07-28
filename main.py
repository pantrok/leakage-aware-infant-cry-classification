"""
Single entry point for the three core experiments of the paper.

    python main.py multiclass --data_dir data/dataset [flags of scripts/optuna_search.py]
    python main.py binary     --data_dir data/dataset [flags of scripts/optuna_binary.py]
    python main.py leakage    --data_dir data/dataset [flags of scripts/leakage_comparison.py]

This is a thin dispatcher: it does not redefine any flags. Everything after
the subcommand is forwarded verbatim to the corresponding script's own
argparse parser (including --help), so the two stay in sync automatically.
Each script also remains runnable on its own, e.g.:

    python scripts/optuna_search.py --data_dir data/dataset ...
"""

import sys

MODES = {
    "multiclass": ("scripts.optuna_search", "optuna_search.py"),
    "binary": ("scripts.optuna_binary", "optuna_binary.py"),
    "leakage": ("scripts.leakage_comparison", "leakage_comparison.py"),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in MODES:
        print(__doc__)
        print(f"Available modes: {', '.join(MODES)}")
        sys.exit(1 if len(sys.argv) >= 2 else 0)

    mode = sys.argv[1]
    forwarded_args = sys.argv[2:]
    module_name, prog_name = MODES[mode]

    import importlib
    target = importlib.import_module(module_name)

    sys.argv = [prog_name] + forwarded_args
    target.main()


if __name__ == "__main__":
    main()
