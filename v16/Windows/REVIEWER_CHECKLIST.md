# Reviewer-requested evidence checklist

This checklist maps the final 174-run suite to the reviewer requests for
manuscript `Access-2026-30971`. It records experimental coverage; the response
letter and manuscript locations are maintained in the private submission
package.

| Request | Released evidence | Status |
|---|---|---|
| R1: characterize the operating boundary | Three-domain main grid and uncapped controls; positive, null, and negative cells retained | Complete |
| R1: resolve implicit/explicit interpretation | Lambda sweep plus separate geometry diagnostic; no task effect attributed to the probe | Complete |
| R1: add formal uncertainty | 47 paired text-cluster comparisons, bootstrap CIs, randomization p-values, Holm correction | Complete |
| R1: clarify decoupled learning rate | Main optimizer and high-learning-rate geometry probe recorded separately | Complete |
| R1: add random signed edges | Medical 10% and legal 600-sentence cells, three seeds each | Complete |
| R1: test lambda and ME weight | Five lambda values and five ME-weight values including the default | Complete |
| R1: downgrade the decision-axis claim | Natural legal benchmark plus explicitly constructed TCM type-edge control | Complete |
| R1: equation numbering | Manuscript-side correction; not an experiment artifact | Complete |
| R1: Chinese herb names | Manuscript and 63-pair online appendix | Complete |
| R1: move the 63-pair list | Separate online appendix in the submission package | Complete |
| R1: justify the 600-sentence budget | Fixed 60/180/600 design plus uncapped comparisons | Complete |
| R1: moderate tone | Boundary-centered interpretation in manuscript and public README | Complete |
| R1: add SGCN/SiGAT | Both adapted to the same prototype graph and NER scorer, three seeds | Complete |
| R1: add uncapped medical/TCM | None/explicit comparisons, three seeds per domain | Complete |
| R2: simplify the abstract | Manuscript-side revision | Complete |
| R2: report P3 variation | Per-seed metrics and sample SD retained in `SUMMARY.json` | Complete |
| R2: explain BERT-CRF across domains | Training support, density, and overlap audits retained in `SUMMARY.json` | Complete |
| R2: verify lambda robustness | Five-value sweep; task predictions unchanged under the tested protocol | Complete |
| R2: qualify CMeIE transfer | Retained as a single-seed directional check only | Complete |
| R2: qualify the theoretical claim | Isolated-edge and fixed-norm assumptions stated in the manuscript | Complete |
| R2: identify CMeEE evaluation | Fixed gold evaluation split distinguished from the hidden leaderboard test | Complete |
| R3: strengthen statistics | Same jointly corrected 47-comparison analysis | Complete |
| R3: separate geometry and task behavior | Probe and explicit task model reported as different experiments | Complete |
| R3: report range and cost | Per-job timing and paired ratios retained | Complete |
| R3: add another Chinese backbone | Chinese RoBERTa-wwm-ext, medical and legal none/explicit cells | Complete |

Technical completion does not guarantee acceptance or establish universal
benefit. Three seeds provide limited evidence about optimization variability,
and normalized-text clusters cannot fully model source-document dependence when
document IDs are unavailable.
