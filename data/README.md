# Expected data layout

The dataset itself is **not** included in this repository (see the main
[README](../README.md#dataset) for why and how to request it). Once you have
access to the Baby Chillanto Infant Cry Database, place it here as:

```
data/
└── dataset/
    ├── 1s_asphyxia/
    ├── 1s_deaf/
    ├── 1s_hunger/
    ├── 1s_normal/
    └── 1s_pain/
```

Each `1s_X` folder contains 1-second `.wav` fragments for class `X`
(`folder_to_class()` in [`src/audio_preprocessing.py`](../src/audio_preprocessing.py)
normalizes folder names to the class label).

All three entry points (`multiclass`, `binary`, `leakage` — see the main
README) run exclusively on these five `1s_*` folders; that is what
reproduces the tables and figures reported in the paper. The original Baby
Chillanto distribution also includes full-length recordings (`Full_X`,
`FullNormal`) that the 1-second fragments were segmented from — those are
not required to run this repository and are not read by default
(`--source 1s`).

Point `--data_dir` at the `dataset/` folder, e.g.:

```bash
python main.py leakage --data_dir data/dataset
```
