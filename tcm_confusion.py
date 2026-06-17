#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Compute the TCM type-confusion matrix.
#
# Trains a no-prior ("none") model on TCM, then tallies a gold-vs-predicted
# type confusion matrix over the test set. Type pairs confused in both
# directions are candidates for a targeted type-level mutex edge.
import os, sys, json
sys.path.insert(0, "run_code")
import torch
from collections import defaultdict
from run_code.ontospan_experiment import (HFEncoder, make_model, read_jsonl, build_types,
                                 load_ontology, run_epoch, set_seed)

set_seed(42)
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", dev)
tr = read_jsonl("data/tcm/train.jsonl")[:600]
dv = read_jsonl("data/tcm/dev.jsonl")[:100]
te = read_jsonl("data/tcm/test.jsonl")[:300]
onto = load_ontology("tcm"); types = build_types(tr + dv, onto)
print("types:", [t for t in types if t != "∅"])
model = make_model(lambda: HFEncoder("hfl/chinese-macbert-base"), types, onto, "none", 10, dev, gnn_depth=2)
optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
for ep in range(3):
    loss = run_epoch(model, tr, optim, dev, rho=5, bs=4, accum=2)
    print(f"  ep{ep+1} loss={loss:.1f}", flush=True)

# For each gold span, record the type the model predicted for the same span.
model.eval()
conf = defaultdict(lambda: defaultdict(int))
with torch.no_grad():
    treps = model.type_representations()
    for text, gold in te:
        reps_l, offs_l = model.encoder([text])
        reps, offs = reps_l[0], offs_l[0]
        sr, cspans = model.span_reps(reps, offs)
        if sr.size(0) == 0:
            continue
        preds = model.score_with(sr, treps).argmax(1).tolist()
        pred_by_span = {cspans[i]: model.types[preds[i]] for i in range(len(cspans))}
        for cs, ce, gt in gold:
            pt = pred_by_span.get((cs, ce), "∅(span>lmax)")
            conf[gt][pt] += 1

real = [t for t in types if t != "∅"]
print("\n=== confusion matrix (row=gold, col=pred; empty=missed) ===")
cols = real + ["∅"]
print("gold/pred".ljust(10) + "".join(f"{c[:5]:>7}" for c in cols))
for gt in real:
    row = conf.get(gt, {})
    tot = sum(row.values()) or 1
    cells = [row.get(c, 0) for c in real] + [row.get("∅", 0) + row.get("∅(span>lmax)", 0)]
    print(f"{gt:<10}" + "".join(f"{c:>7}" for c in cells) + f"   (n={tot})")

print("\n=== top type-confusion pairs (gold->pred, excluding misses) ===")
pairs = []
for gt in real:
    for pt, c in conf.get(gt, {}).items():
        if pt in real and pt != gt and c > 0:
            pairs.append((c, gt, pt))
pairs.sort(reverse=True)
for c, gt, pt in pairs[:10]:
    print(f"  {gt} → {pt}: {c}")
print("\nreading: large bidirectional pairs are genuine confusions -> targeted type-mutex candidates.")
json.dump({f"{gt}|{pt}": c for c, gt, pt in pairs},
          open("tcm_confusion.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("saved tcm_confusion.json")
