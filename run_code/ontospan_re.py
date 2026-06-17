#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Relation-extraction model (transfer experiment).
#
# Data and model utilities for the CMeIE relation-extraction transfer experiment
# (a signed graph over relation types).
import os, json, argparse
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RE_DIR = os.path.join(_ROOT, "data", "re", "CMeIE-V2")
NA = "NA"


def read_schema(path):
    txt = "".join(l for l in open(path, encoding="utf-8") if not l.lstrip().startswith("#"))
    return json.loads(txt)


def build_relation_graph(schema):
    preds = [s["predicate"] for s in schema]
    relations = [NA] + preds
    r2i = {r: i for i, r in enumerate(relations)}
    by_sig = defaultdict(list)
    by_subj = defaultdict(list)
    for s in schema:
        by_sig[(s["subject_type"], s["object_type"])].append(s["predicate"])
        by_subj[s["subject_type"]].append(s["predicate"])
    mutex, seen = [], set()
    for sig, ps in by_sig.items():
        for a in range(len(ps)):
            for b in range(a + 1, len(ps)):
                key = frozenset((ps[a], ps[b]))
                if key in seen: continue
                seen.add(key); mutex.append((r2i[ps[a]], r2i[ps[b]]))
    mset = {frozenset(p) for p in mutex}
    attract, seen_a = [], set()
    for st, ps in by_subj.items():
        u = sorted(set(ps))
        for a in range(len(u)):
            for b in range(a + 1, len(u)):
                key = frozenset((u[a], u[b]))
                if key in seen_a or key in mset: continue
                seen_a.add(key); attract.append((r2i[u[a]], r2i[u[b]]))
    return relations, attract, mutex


def _ovalue(x):
    return x.get("@value") if isinstance(x, dict) else x


def load_cmeie(path, max_lines=None):
    data, miss, total = [], 0, 0
    for ln in open(path, encoding="utf-8"):
        ln = ln.strip()
        if not ln: continue
        if max_lines and len(data) >= max_lines: break
        o = json.loads(ln)
        text = o["text"]
        ent_idx = {}
        ents, triples = [], []
        def add_ent(name, etype):
            cs = text.find(name)
            if cs < 0: return None
            ce = cs + len(name)
            k = (cs, ce, etype)
            if k not in ent_idx:
                ent_idx[k] = len(ents); ents.append((cs, ce, etype, name))
            return ent_idx[k]
        for spo in o.get("spo_list", []):
            total += 1
            subj = spo["subject"]; obj = _ovalue(spo["object"])
            st = spo.get("subject_type"); ot = _ovalue(spo.get("object_type"))
            si = add_ent(subj, st); oi = add_ent(obj, ot)
            if si is None or oi is None:
                miss += 1; continue
            triples.append((si, oi, spo["predicate"]))
        data.append({"text": text, "entities": ents, "triples": triples})
    return data, {"sentences": len(data), "spo_total": total, "spo_unlocatable": miss}


def load_re(caps=(None, None, None), max_check=None):
    schema = read_schema(os.path.join(RE_DIR, "53_schemas.json"))
    relations, attract, mutex = build_relation_graph(schema)
    tr, st_tr = load_cmeie(os.path.join(RE_DIR, "CMeIE-V2_train.jsonl"), caps[0] or max_check)
    dv, st_dv = load_cmeie(os.path.join(RE_DIR, "CMeIE-V2_dev.jsonl"), caps[1] or max_check)
    return tr, dv, relations, attract, mutex, {"train": st_tr, "dev": st_dv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--max", type=int, default=None, help="read only the first N sentences (self-check)")
    a = ap.parse_args()
    schema = read_schema(os.path.join(RE_DIR, "53_schemas.json"))
    relations, attract, mutex = build_relation_graph(schema)
    print(f"relations (incl. NA)={len(relations)} | mutex edges={len(mutex)} | attract edges={len(attract)}")
    if a.check:
        tr, st = load_cmeie(os.path.join(RE_DIR, "CMeIE-V2_train.jsonl"), a.max or 2000)
        n_ent = sum(len(d["entities"]) for d in tr)
        n_tri = sum(len(d["triples"]) for d in tr)
        rel_count = defaultdict(int)
        for d in tr:
            for _, _, p in d["triples"]: rel_count[p] += 1
        print(f"\n[train sample {st['sentences']} sents] entities={n_ent} triples={n_tri} "
              f"| spo total={st['spo_total']} unlocatable={st['spo_unlocatable']} "
              f"({100*st['spo_unlocatable']/max(1,st['spo_total']):.1f}%)")
        print("  per-sentence avg: entities %.1f, triples %.1f" % (n_ent/max(1,st['sentences']), n_tri/max(1,st['sentences'])))
        d0 = next(d for d in tr if d["triples"])
        print("\n  sample:", d0["text"][:60])
        print("   entities:", d0["entities"][:4])
        print("   triples:", [(d0["entities"][si][3], p, d0["entities"][oi][3]) for si, oi, p in d0["triples"][:4]])
        print("\n  mutex-edge examples (relations competing for the same slot):",
              [(relations[i], relations[j]) for i, j in mutex[:5]])


if __name__ == "__main__":
    main()
