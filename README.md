# OntoSpan

**Dual-layer signed ontology constraints for low-resource Chinese named entity recognition.**

OntoSpan is a span-based NER framework for testing whether documented exclusion
knowledge changes predictions in specialized Chinese domains. It represents
compatible and mutually exclusive relations on a two-layer signed graph: a
type layer for label-level relations and an instance layer for domain concepts.
The current study covers legal text, Traditional Chinese Medicine (TCM), and
medical NER.

This repository contains the experiment code and result artifacts. The
licensed benchmark texts and the manuscript are not redistributed.

## Revision experiment status

The reviewer-driven experiment suite is complete: **174/174 planned
configurations**, with no missing or invalid runs. Every reported cell uses the
same frozen data policy and three seeds (`42`, `1`, and `2`). The 47
prespecified paired comparisons were analyzed jointly with text-cluster
bootstrap confidence intervals, paired randomization tests, and Holm
correction.

The expanded evidence does **not** support a universal low-resource gain. The
effect depends on domain, supervision budget, graph construction, and encoder.
For the capped study, training sizes are 60, 180, and 600 sentences; development
and fixed gold evaluation sets contain 100 and 300 sentences, respectively.

| Domain | 60 sentences | 180 sentences | 600 sentences |
|---|---:|---:|---:|
| TCMNER | +0.160 | -0.638 | +2.987 |
| CAIL | -2.307 | -2.807 | +2.991 |
| CMeEE | +0.381 | +2.090 | +0.817 |

Values are explicit-minus-no-ontology changes in span-level micro-F1 points,
averaged over three seeds. Uncertainty and adjusted p-values are reported in
[`result/revision_20260914/STATISTICS.json`](result/revision_20260914/STATISTICS.json).

### Reviewer-requested additions

- **Random-sign control.** Graph topology and positive/negative edge counts are
  held fixed while sign identities are shuffled. Random signs outperformed the
  documented signs in the tested legal cell and also had a higher medical mean,
  so the gains cannot be attributed uniquely to the semantic identity of the
  supplied signs.
- **Signed-GNN baselines.** SGCN and SiGAT were adapted to the same ontology
  graph, encoder, span scorer, data, and seed protocol. They were competitive
  with or stronger than OntoSpan in the tested cells.
- **Second Chinese backbone.** With Chinese RoBERTa-wwm-ext, the positive legal
  MacBERT mean did not transfer; the medical difference was positive but did not
  survive the joint correction.
- **Sensitivity analysis.** The implicit realization was prediction-invariant
  for `lambda` in `{0, 0.05, 0.1, 0.5, 1.0}`. The explicit negative-edge weight
  sweep was non-monotonic and did not reveal a stable alternative optimum.
- **Uncapped training.** Three-seed comparisons were added for medical and TCM
  alongside the existing legal full-corpus check. The medical difference was
  +0.110 F1 points; the TCM mean difference was +0.664 points but varied
  substantially across seeds.
- **Type-level control.** An exploratory TCM cause-versus-symptom type edge was
  added to probe the decision-axis hypothesis. It is a constructed control, not
  a second natural type-level ontology benchmark.
- **Cost and uncertainty.** Training time, per-seed variation, paired confidence
  intervals, raw p-values, and Holm-adjusted p-values are retained even when
  they weaken the original interpretation.

The implicit signed-Laplacian objective is retained as a geometric diagnostic:
under a separate high-learning-rate probe it moved mutex pairs toward
antipodal geometry, but it did not change task predictions under the tested
training protocol. That probe is not treated as an explanation of the explicit
model's task-level F1 changes.

## Result artifacts

The final revision evidence is under
[`result/revision_20260914/`](result/revision_20260914/):

- `MANIFEST.json` — the exact 174-job configuration manifest;
- `SUMMARY.json` — per-run metrics, per-type scores, data/graph audits, and
  training times, without benchmark text;
- `STATISTICS.json` — all 47 prespecified paired comparisons, confidence
  intervals, raw p-values, and Holm-adjusted p-values;
- `README.md` — a compact guide to the released evidence and its limitations.

Earlier result directories are retained for provenance. Use the dated revision
directory for the current paper numbers.

## Repository layout

```text
build_datasets.py               entity dictionaries and signed ontologies
run_code/                       shared model, training, and evaluation code
  ontospan_experiment.py        OntoSpan, random-sign control, SGCN, SiGAT
  run_ablation.py               training and result serialization
  run_grid_safe.py              isolated experiment-grid runner
v16/Windows/                    frozen 174-configuration revision suite
result/
  revision_20260914/            final aggregate revision evidence
  results_3seed/                earlier main medical/TCM results
  results_law/                  earlier capped legal results
  results_law_full/             earlier uncapped legal result
  results_p3_*/                 real/rewired/removed perturbations
  results_baselines/            traditional and LLM baselines
  figures/                      publication figures in PNG/SVG
```

## Reproducing the final suite

The exact final 174-job protocol is frozen in `v16/Windows`. On a Windows CUDA
machine:

```powershell
cd v16\Windows
python 01_SETUP_WINDOWS.py
python 02_CHECK_ENVIRONMENT.py
python 06_ALL_174.py
python 07_SUMMARIZE.py
```

The launcher validates code, model, configuration, and data fingerprints before
reusing a completed job. It stores model weights locally but excludes them from
the summary archive. The core modules in `run_code/` remain platform-neutral
and can also be invoked through the earlier root runners.

Developer tests for the frozen suite can be run without training the full grid:

```bash
python -m unittest discover -s v16/Windows -p "test_*.py" -v
```

## Data and evaluation

The experiments use the public TCMNER, CMeEE/CBLUE, CMeIE/CBLUE, and CAIL
benchmarks under their respective licenses. Benchmark texts are not committed;
obtain them from their official sources and place them under `data/` as
described in `run_code/download_datasets.py`.

NER scores use exact-match span-level micro precision, recall, and F1. CMeEE is
evaluated on the fixed gold split used by this study; it is distinct from the
hidden CBLUE leaderboard test, and no direct leaderboard comparison is made.

## Scope

This release supports reproduction and audit of the reported experiments. It
does not establish that signed ontology constraints improve every domain,
budget, graph, or encoder. The negative and null outcomes are part of the
release rather than filtered out.

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
