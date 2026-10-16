# Pre-specification: RoBERTa-wwm-ext on the two remaining cells (written before any of these runs)

Cells: medical 600 sentences, TCM 600 sentences; primary protocol; seeds 42, 1 to 9; five configurations
(none, explicit, random sign, ME removed, SGCN); 100 runs.

Question: does the legal-cell contrast explicit - random sign (+4.0, Holm p = 0.031 over 8 comparisons) recur?

Decision rule, fixed now: the contrast is reported as replicated only if explicit - random sign is positive with the
bootstrap interval excluding zero in at least one of the two new cells and the sign is not reversed in the other.
Otherwise it is reported as not replicated. All 8 new comparisons (4 per cell x 2 cells) plus the 8 earlier
are corrected together with Holm (16 comparisons). Whatever the result, it is reported in the manuscript.
