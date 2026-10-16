# Pre-registration: controlled synthetic check of the ME mechanism
Written 2026-10-15, before any arm comparison on synthetic data was run.

## Purpose
Real ontologies give the ME constraint almost nothing to act on (law: one live edge; medical: none).
This check asks: in a setting built so that ME *should* matter, does it?  It tests the mechanism, not real-task benefit.

## Design (fixed)
- 8 types: two ME pairs (甲/乙, 丙/丁; confusable, `k` training sentences each), one related pair (戊/己), two easy types.
- Confusability: prefix cue uninformative with prob 0.6; tail cue informative with prob 0.7 (~18% irreducible ambiguity).
- `k` in {10, 40}. Test: 40 sentences per ME type. Every replicate is an independent dataset (data seed = training seed).
- Arms: none | implicit (lam 0.1) | explicit(real signs) | explicit(random_sign) | explicit(removed).
- Seeds 42,1,2,3,4,5,6,7,8,9. Two optimisation protocols: default lr (type embeddings at 2e-5) and type-embedding lr 1e-3.
- Model selection by dev only. Unmodified v16 worker.train().

## Difficulty calibration (the only permitted adjustment)
A 2-seed pilot of the `none` arm only (no arm comparison) checks that the task is neither at ceiling (ME-type F1 > 0.95)
nor at floor (< 0.40). If it is, prefix/tail probabilities may be changed once, and the change is reported.

## Primary metric and decision rule
- Primary metric: micro-F1 over the four ME types (tp/fp/fn summed).  Secondary: overall micro-F1.
- Comparisons (paired by seed; exact sign-flip test; 95% bootstrap CI; Holm over all 16):
  explicit-none, explicit-randsign, explicit-removed, implicit-none  x  k in {10,40}  x  2 protocols.
- "The mechanism works when it should" is claimed only if explicit beats none AND randsign AND removed in the same
  (k, protocol) cell with Holm-adjusted p < 0.05.  Otherwise the result is reported as "not detected".
- No arm, parameter or metric is added after seeing results; anything further is labelled exploratory.

## Calibration log (appended before the main grid)
- Pilot v1 (prefix-uninformative 0.6, tail-informative 0.7; `none` arm, seeds 42 and 1): ME-type F1 = 0.310, 0.277 (< 0.40 floor).
- One permitted change: prefix-uninformative 0.3, tail-informative 0.9.
- Pilot v2: ME-type F1 = 0.427, 0.382 (mean 0.404). Accepted; no further change. Only the `none` arm was run in pilots.
- The two pilot seeds are included in the main grid (deterministic, identical data).
