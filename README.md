# OntoSpan

**Signed ontology constraints for low-resource Chinese NER.**

OntoSpan is a span-based Chinese named-entity recognizer that treats ontology
relations as *polarity-aware* constraints on a two-layer signed graph:
semantically compatible relations are smoothed, while **mutually exclusive
(mutex)** relations are *anti-smoothed* toward antipodal separation, under a
single signed-Laplacian objective. The mutex constraints come from
authoritative external sources (pharmacopoeial incompatibility lists, statutory
role exclusivity), not from benchmark labels.

This repository contains the **experiment code and result files** for the
study. The manuscript and the licensed benchmark corpora are not included
(see *Data* and *Scope* below).

## Key finding: a decision-axis principle

The value of an exclusion prior is governed by whether the mutex lies on the
entity-typing decision axis.

- **Type-level** exclusivity (e.g. legal *suspect* ⊥ *victim*) lies on that
  axis and yields a durable gain that *grows* with supervision (up to **+4.4
  F1** at full data on legal NER, positive across all seeds).
- **Instance-level** mutex (e.g. contraindicated herb/drug pairs) is orthogonal
  to type decisions: it helps only under extreme scarcity (where on medical NER
  it beats biaffine, global-pointer, W2NER, and a collapsing BERT-CRF) and is
  otherwise absorbed by supervision.
- The signed objective provably induces the predicted antipodal geometry
  (mutex-pair cosine **−0.64 to −0.99**, vs. +0.85 under the raw encoder —
  induced, not inherited).

## Repository layout

```
build_datasets.py        Entity dictionaries + signed ontology
                         (mutex / attractive edges, per domain)
run.py                   Unified entry point (law-small / law-full /
                         medical-full / all)
run_code/                Core library
  ontospan_experiment.py   Model (SignedGAT), training, evaluation
  baselines.py             biaffine / global-pointer / BERT-CRF / W2NER
  run_grid_safe.py         Subprocess-isolated grid driver (avoids OOM)
  run_ablation.py          Section-style ablations
  ontospan_re*.py          Relation-extraction transfer (CMeIE)
  download_datasets.py     Fetch + convert the three benchmarks
result/                  All results (read-only artifacts)
  results_3seed/           Main table: TCM + medical, none/explicit, 3 seeds
  results_law/             Legal NER (type-level mutex), 3 seeds
  results_baselines/       Four external baselines, per domain
  results_p3_*/            Perturbation controls: real / rewired / removed
  results_re/              Relation-extraction negative-transfer control
  results_tcm_biaosym_3seed/  Within-TCM type-mutex control (cause ⊥ symptom)
  P1_cosine_implicit.json  Geometry: mutex-pair cosines (P1)
  tcm_confusion.json       TCM type-confusion counts
  figures/                 Publication figures (PNG)
  RESULTS.txt              Plain-text result summary
```

### Experiment runner scripts (root)

The runner scripts share one backbone, budget, and protocol so that
comparisons reflect architecture rather than tuning. The `_4070` suffix only
records that runs were performed on an RTX 4070; the scripts are CUDA-agnostic.

| Script | What it runs |
|---|---|
| `run_4070.py` | Main table (multi-seed) + perturbation (P3) + geometry (P1) |
| `run_fast_4070.py` | Parallel grid driver (saturates one GPU) |
| `run_baselines_4070.py` | The four external baselines (merge-writes per domain) |
| `run_law_4070.py` | Legal NER — type-level mutex (CAIL) + P1 |
| `run_re_4070.py` | Relation-extraction transfer grid (CMeIE) |
| `run_A_tcm.py` | TCM ablation: routing instance anchors into scoring |
| `run_A2_typemutex.py` | TCM with an on-axis type-level mutex (cause ⊥ symptom) |
| `verify_p1_cosine.py` | Geometry check: mutex-pair representation cosine (P1) |
| `tcm_confusion.py` | TCM type-confusion matrix |
| `baseline_llm_deepseek.py` | Few-shot LLM baseline (DeepSeek; API, no GPU) |
| `collect_results.py` | Bundle `result/` artifacts into a zip |

## Reproducing

```bash
pip install -r requirements.txt
# 1) fetch the benchmarks into data/<domain>/ (see run_code/download_datasets.py)
python run_code/download_datasets.py
# 2) main grid + perturbation + geometry
python run_4070.py
# 3) external baselines (same backbone / budget)
python run_baselines_4070.py
# 4) legal type-level experiment
python run_law_4070.py
```

Results are written under `result/`; `result/RESULTS.txt` and
`run_code/RESULTS_README.md` describe which directory is the source of each
table.

## Data

All experiments use **public** Chinese benchmarks — TCMNER, CMeEE and CMeIE
(CBLUE), and the CAIL legal information-extraction task — under their
respective licences. These corpora are **not redistributed here**; obtain them
from their official sources and place them under `data/`. The external ontology
resources (incompatibility lists, statutory role-exclusivity pairs) are
compiled from publicly citable sources and built in `build_datasets.py`.

## Scope

This release contains code and result artifacts only; the manuscript is under
peer review and is omitted. The code is provided in anonymized form for
double-blind review and will be linked from the final paper upon acceptance.

## License

Released under the MIT License (see `LICENSE`).
