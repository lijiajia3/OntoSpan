# run_code/ — core experiment library

Importable/runnable modules behind the top-level runner scripts. Each module
adds the project root to `sys.path` and resolves data from `<root>/data/`, so it
can be launched from any working directory (`python run_code/<file>.py`). The
shared two-layer ontology source (`MUTEX_TYPE` / `MUTEX_INST` / `ATTRACT`) lives
in the root `build_datasets.py` and is imported by many modules.

| File | Purpose |
|---|---|
| `ontospan_experiment.py` | Model engine: encoder → span enumeration → two-layer signed graph → bilinear scoring; data loading, training, evaluation. Modes `none` / `implicit` / `explicit`. |
| `baselines.py` | External baselines: biaffine, global-pointer, BERT-CRF, W2NER (shared backbone/budget). |
| `run_grid_safe.py` | Subprocess-isolated grid driver (one config per process; avoids GPU memory accumulation). Supports `--type-mutex` injection. |
| `run_ablation.py` | Section-style ablation + baseline comparison; per-epoch P/R/F1, ΔF1 vs. baseline, per-type F1. |
| `ontospan_re.py`, `ontospan_re_train.py` | Relation-extraction transfer (CMeIE), used as a negative-transfer control. |
| `download_datasets.py` | Download + normalize the three benchmarks into a unified span-NER format under `data/<domain>/{train,dev,test}.jsonl`. |
| `RESULTS_README.md` | Which `result/` directory is the source of each table. |

## Realizations (paper Section 3)

| Mode | Mechanism | Inference cost |
|---|---|---|
| `none` | Baseline span-NER, no ontology | — |
| `implicit` | Signed-graph Laplacian regularizer: attract `w⁺‖z_i−z_j‖²`, mutex `w⁻‖z_i+z_j‖²` | zero overhead |
| `explicit` | Mutex-aware graph attention: attract neighbors aggregated positively, mutex neighbors negatively, with a gated residual; `--gnn-depth` controls hops | GNN pass at forward time |

The default encoder is `hfl/chinese-macbert-base`. Dataset licences differ
(e.g. CMeEE requires authorization); obtain each from its official source.
