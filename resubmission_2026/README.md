# Resubmission materials (IEEE Access, Access-2026-46722)

This folder holds the scripts, analysis plans, and per-run results added for the resubmission. The original
implementation in `v16/Windows` is unchanged; every script here wraps its `worker.train()` and edits no repository file.

## Layout

| Folder | Content |
|---|---|
| `scripts/` | Experiment grids, diagnostics, and analysis scripts. Copy them into `v16/Windows` before running. |
| `plans/` | Analysis plans written before the corresponding runs (synthetic check, second RoBERTa batch, separation control). |
| `results/` | Per-run metrics for every run reported in the revised manuscript. |

## Scripts

| File | Manuscript section | Purpose |
|---|---|---|
| `big_grid.py` | VI-B | Ten-seed grid: legal 600, medical 180 and 600, TCM 600 × {none, explicit, random sign, ME removed, SGCN} × seeds 42 and 1 to 9. `BIG_LR_TYPE=0.001` selects the second protocol (type embeddings and concept anchors at 1e-3). |
| `analyze_big.py` | VI-B, VI-C | Paired seed-level analysis: bootstrap interval over seeds, exact sign-flip test, Holm correction, noise floor. |
| `roberta_grid.py` | VI-B | Second encoder (RoBERTa-wwm-ext), primary protocol. The first batch ran legal 600 and medical 180; the second batch ran the same script with `CELLS = (('medical', 1.0), ('tcm', 1.0))`. |
| `diag_lambda_law.py`, `summarize_diag.py` | VI-E | Implicit-weight sweep on the legal cell (λ ∈ {0, 0.1, 1}, seeds 42, 1, 2, both protocols), logging the change of the type embeddings, the regularizer, the prototype cosine, and prediction differences. |
| `sep_control.py` | VI-E | Separation control: implicit term, type embeddings at 1e-3, λ ∈ {0, 1, 10}, ten seeds. |
| `me_geometry.py` | VI-E | Reads the stored geometry of every legal run and reports the suspect/victim prototype cosine after the graph module. |
| `synth_grid.py`, `analyze_synth.py`, `chain_synth2.sh` | VI-F | Controlled synthetic check. `chain_synth2.sh` restarts the grid because the host process was killed by the out-of-memory killer about every 13 runs. |
| `dump_all.py`, `dump_diag.py` | – | Export the run directories to the JSON files in `results/`. |

## Results

| File | Runs | Content |
|---|---:|---|
| `all_runs.json` | 800 | Ten-seed grid under both protocols (400), RoBERTa (200), synthetic check (200). Test metrics, true/false positives, false negatives, per-type counts, best epoch, and dev score. |
| `lambda_sweep_and_separation_runs.json` | 42 | λ sweep and separation control, with the change of the type embeddings, regularizer values, prototype cosines, and the number of test sentences whose predictions differ from λ = 0. |
| `previous_machine_runs_rtx4070.json` | 174 | Runs of the previous version (RTX 4070, Windows), used to recompute the three-seed differences reported earlier. |
| `me_geometry_law.json` | – | Summary of the suspect/victim prototype cosine by configuration (initial and selected epoch). |

## Environment

Ubuntu, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), PyTorch 2.9.1 (CUDA 12.8), fp32. Encoders
`hfl/chinese-macbert-base` (revision `a986e004d2a7f2a1c2f5a3edef4e20604a974ed1`) and `hfl/chinese-roberta-wwm-ext`
(revision `5c58d0b8ec1d9014354d691c538661bf00bfdb44`). Runs repeat exactly on one machine; results differ across
hardware and library builds (manuscript, Section VI-C).
