# Analysis plan: does realized prototype separation help? (written 2026-10-15, before any of these runs)

Motivation: the stored geometry shows that the explicit operator does not separate the suspect/victim prototypes
(cosine rises from +0.01 to +0.08). The implicit signed-Laplacian term acts directly on the type embeddings and, with a
decoupled embedding learning rate of 1e-3, did push the pair apart in the three-seed sweep. This control asks whether
separation, once it is actually realized, changes legal F1.

Design: legal 600-sentence cell, implicit mode, type embeddings and anchors at 1e-3, all else as in the primary protocol;
lambda in {0, 1, 10}; seeds 42, 1 to 9 (30 runs; the six runs at lambda 0 and 1 for seeds 42, 1, 2 are reused from the
earlier sweep, identical code).

Primary comparisons (2): lambda=1 minus lambda=0 and lambda=10 minus lambda=0, test F1, paired by seed; percentile
bootstrap interval over seeds, exact sign-flip test, Holm over the 2 comparisons.
Manipulation check: cosine of the suspect/victim type embeddings at the selected epoch must be lower than at lambda=0;
if it is not, the control is reported as failed rather than as evidence.
All results are reported whatever their direction.
