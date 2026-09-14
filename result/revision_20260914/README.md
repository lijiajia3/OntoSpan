# Final reviewer-driven experiment release (2026-09-14)

This directory is the compact public record for the revised OntoSpan study.
It contains aggregate evidence only: no benchmark sentences, model weights,
credentials, or machine-specific paths are included.

## Completion and protocol

- Planned configurations: **174**
- Completed and validated: **174**
- Missing or invalid: **0**
- Seeds: `42`, `1`, `2`
- Main capped training sizes: `60`, `180`, `600` sentences
- Capped development/evaluation sizes: `100` / `300` sentences
- Main encoder: `hfl/chinese-macbert-base`
- Second encoder: `hfl/chinese-roberta-wwm-ext`
- Metric: exact-match span-level micro-F1
- Paired inference: 2,000 text-cluster bootstrap/randomization replicates
- Multiplicity control: Holm correction across 47 prespecified comparisons

The paired tests quantify uncertainty over the fixed evaluation texts for the
fitted models. They are not a substitute for a large population of independent
training runs; the three-seed standard deviations should be read alongside the
paired intervals and p-values.

## Main capped results

The table reports mean F1 and the explicit-minus-none change in F1 points.

| Domain | Training sentences | None | Explicit | Change |
|---|---:|---:|---:|---:|
| TCMNER | 60 | 0.229 | 0.230 | +0.160 |
| TCMNER | 180 | 0.297 | 0.291 | -0.638 |
| TCMNER | 600 | 0.318 | 0.348 | +2.987 |
| CAIL | 60 | 0.226 | 0.203 | -2.307 |
| CAIL | 180 | 0.275 | 0.247 | -2.807 |
| CAIL | 600 | 0.360 | 0.390 | +2.991 |
| CMeEE | 60 | 0.123 | 0.127 | +0.381 |
| CMeEE | 180 | 0.139 | 0.160 | +2.090 |
| CMeEE | 600 | 0.250 | 0.259 | +0.817 |

These outcomes define a mixed operating boundary. They do not support a claim
that the signed prior is uniformly beneficial at the smallest budget or that
its benefit grows monotonically with supervision.

## Added controls and experiments

| Experiment | Configurations represented | Main observation |
|---|---:|---|
| Real / rewired / removed edges | 30 | Ordering changes across domains and budgets; several contrasts do not survive joint correction. |
| Random sign identities | 6 | Random signs exceed the documented signs in the tested legal cell and have a higher medical mean. |
| Chinese RoBERTa-wwm-ext | 12 | The positive legal MacBERT mean does not transfer to the second encoder. |
| SGCN and SiGAT adaptations | 12 | Established signed operators are competitive or stronger in the tested cells. |
| Implicit `lambda` sweep | 15 | All five values yield identical task predictions under the tested protocol. |
| Explicit ME-weight sweep | 12 | The response is non-monotonic; no stable alternative optimum is established. |
| Uncapped medical and TCM | 12 | Medical is nearly unchanged; TCM's positive mean varies strongly across seeds. |
| Constructed TCM type edge | 9 | Supports further testing of the decision-axis hypothesis but is not an independent natural benchmark. |
| Traditional medical baselines | 12 | Biaffine, GlobalPointer, BERT-CRF, and W2NER use the same data/encoder budget. |

Training time is included in `SUMMARY.json`. Across the capped medical budgets,
the explicit-to-none time ratio is approximately 1.43--1.48; for the legal
600-sentence cell it is 2.81.

## Files

- [`MANIFEST.json`](MANIFEST.json) records every planned job and its frozen
  configuration.
- [`SUMMARY.json`](SUMMARY.json) records validated per-run metrics, seed-level
  results, per-type metrics, support/coverage audits, graph summaries, and
  runtime measurements.
- [`STATISTICS.json`](STATISTICS.json) contains the 47 prespecified paired
  effects, bootstrap intervals, raw randomization p-values, and Holm-adjusted
  p-values.

## Interpretation boundary

The implicit high-learning-rate geometry probe and the explicit task model are
different experiments. The former shows how the isolated signed objective can
reshape ontology embeddings under its probe settings; it does not explain the
explicit model's task-level changes. Likewise, fixed-model paired significance
does not establish consistency across unseen training seeds.

CMeEE uses the study's fixed gold evaluation split rather than the hidden CBLUE
leaderboard test. The released metrics cover every evaluation sentence in that
split under the stated exact-match NER protocol.
