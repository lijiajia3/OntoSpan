# -*- coding: utf-8 -*-
# Download and normalize the benchmark datasets.
#
# Fetches and normalizes TCMNER / CMeEE / CAIL into a unified span-NER JSONL format
# under data/<domain>/. Some sources are licence-gated and require manual download.

import os, re, json, glob, argparse, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(_ROOT, "data")
RAW_ROOT = os.path.join(_ROOT, "data", "raw")

CMEEE_LABEL_MAP = {
    "dis": "疾病", "sym": "临床表现", "dru": "药物", "equ": "医疗设备",
    "pro": "医疗程序", "bod": "身体部位", "ite": "检验项目", "dep": "科室", "mic": "微生物",
    "疾病": "疾病", "临床表现": "临床表现", "药物": "药物", "医疗设备": "医疗设备",
    "医疗程序": "医疗程序", "身体": "身体部位", "身体部位": "身体部位",
    "医学检验项目": "检验项目", "检验项目": "检验项目", "科室": "科室",
    "微生物类": "微生物", "微生物": "微生物",
}
CAIL_LABEL_MAP = {
    "NHCS": "犯罪嫌疑人", "NHVI": "受害人", "NCSM": "被盗货币", "NCGV": "物品价值",
    "NCSP": "盗窃获利", "NASI": "被盗物品", "NATS": "作案工具", "NT": "时间",
    "NS": "地点", "NO": "组织机构",
    "犯罪嫌疑人": "犯罪嫌疑人", "受害人": "受害人", "被盗货币": "被盗货币",
    "物品价值": "物品价值", "盗窃获利": "盗窃获利", "被盗物品": "被盗物品",
    "作案工具": "作案工具", "时间": "时间", "地点": "地点", "组织机构": "组织机构",
}

TCMNER_LABEL_MAP = {
    "HER": "中药", "SYM": "症状", "PRE": "方剂", "CAU": "病因", "EFF": "功效", "Person": "人物",
    "中药": "中药", "中药材": "中药", "herb": "中药", "herbs": "中药", "drug": "中药",
    "中成药": "方剂", "preparation": "方剂", "preparations": "方剂",
    "症状": "症状", "symptom": "症状", "病因": "病因", "cause": "病因",
    "功效": "功效", "effect": "功效", "efficacy": "功效",
}

SOURCES = {
    "cmeee": {
        "domain": "medical",
        "hf_candidates": ["Rosenberg/CMeEE", "Aunderline/CMeEE", "Rosenberg/CMeEE-V2"],
        "label_map": CMEEE_LABEL_MAP,
        "official": "Tianchi CBLUE: https://tianchi.aliyun.com/dataset/144495 (registration + agreement required)",
        "manual": ("(1) register on Tianchi and download CMeEE_train.json / CMeEE_dev.json (test has no labels);"
                   "(2) place it under data/raw/cmeee/ and rerun this script."),
        "raw_glob": "*.json",
    },
    "cail": {
        "domain": "law",
        "hf_candidates": [],
        "label_map": CAIL_LABEL_MAP,
        "official": ("CAIL judicial-text IE (xxcq): registration http://cail.cipsc.org.cn/ (IE track)."
                     "the official GitHub (china-ai-law-challenge/CAIL2021) has only code + samples; the real training data needs registration"
                     "(licence-gated like CMeEE; no legal free mirror, so not auto-downloaded)."),
        "manual": ("(1) register and download the IE train.json (format {context, entities:[{label, span:['s;e',...]}]});"
                   "(2) place under data/raw/cail/ and rerun -- the CAIL parser is built in (NHCS->suspect / NHVI->victim, 10 types,"
                   "auto-normalized to span char offsets; adjust CAIL_SPAN_INCLUSIVE for off-by-one. BIO conll / jsonl-spans also supported."),
        "raw_glob": "*",
    },
    "tcmner": {
        "domain": "tcm",
        "hf_candidates": [],
        "github_raw": [
            "https://raw.githubusercontent.com/cshan-github/TCM_NER_datasets/main/train.txt",
            "https://raw.githubusercontent.com/cshan-github/TCM_NER_datasets/main/test.txt",
            "https://raw.githubusercontent.com/cshan-github/TCM_NER_datasets/master/train.txt",
            "https://raw.githubusercontent.com/cshan-github/TCM_NER_datasets/master/test.txt",
        ],
        "label_map": TCMNER_LABEL_MAP,
        "official": ("integrated: cshan-github/TCM_NER_datasets (GitHub, BIO; symptom/cause/herb/formula/efficacy)."
                     "alternative: TCM package-insert NER (13 types, Tianchi/Kaggle, each licence-gated);"
                     "or use --hf-id for another verified TCM NER dataset."),
        "manual": ("by default auto-fetch train/test.txt from cshan GitHub; on failure, manually place files under data/raw/tcmner/ "
                   "(BIO conll / jsonl-spans supported)."),
        "raw_glob": "*",
    },
}


def _map_label(label, label_map):
    if label_map is None:
        return label
    return label_map.get(label, label_map.get(str(label).lower(), label))


def from_hf_entities_example(ex, label_map):
    text = ex.get("text") or ex.get("sentence") or ex.get("context") or ""
    ents = ex.get("entities") or ex.get("entity") or ex.get("ner") or []
    out = []
    for e in ents:
        if not isinstance(e, dict):
            continue
        s = e.get("start_idx", e.get("start", e.get("begin")))
        t = e.get("end_idx",  e.get("end",  e.get("stop")))
        lab = e.get("type", e.get("label", e.get("entity_type")))
        if s is None or t is None or lab is None:
            continue
        s = int(s); t = int(t)
        end = t + 1 if t < len(text) and (t + 1 - s) == len(e.get("entity", e.get("text", "") ) or "") else (t + 1)
        surf = e.get("entity", e.get("text"))
        if surf:
            if text[s:s + len(surf)] == surf:
                end = s + len(surf)
        end = max(end, s + 1)
        out.append([s, min(end, len(text)), _map_label(lab, label_map)])
    return text, out


def from_bio_lines(lines, label_map):
    sents, toks, tags = [], [], []
    def flush():
        if not toks:
            return
        text = "".join(toks); ents = []; i = 0
        while i < len(tags):
            tag = tags[i]
            if tag and tag[0] in "BS" and "-" in tag:
                lab = tag.split("-", 1)[1]; j = i + 1
                while j < len(tags) and tags[j][:2] in ("I-", "E-") and tags[j].split("-", 1)[1] == lab:
                    j += 1
                ents.append([i, j, _map_label(lab, label_map)]); i = j
            else:
                i += 1
        sents.append((text, ents)); toks.clear(); tags.clear()
    for ln in lines:
        ln = ln.rstrip("\n")
        if not ln.strip():
            flush(); continue
        if "\t" in ln:
            parts = ln.split("\t")
        elif "," in ln and " " not in ln:
            parts = ln.rsplit(",", 1)
        else:
            parts = ln.split()
        if len(parts) >= 2 and parts[0] != "":
            toks.append(parts[0]); tags.append(parts[-1].strip())
    flush()
    return sents


def from_jsonl_spans(obj, label_map):
    text = obj.get("text") or obj.get("sentence") or ""
    raw = obj.get("entities") or obj.get("label") or obj.get("spans") or []
    out = []
    for e in raw:
        if isinstance(e, (list, tuple)) and len(e) >= 3:
            out.append([int(e[0]), int(e[1]), _map_label(e[2], label_map)])
        elif isinstance(e, dict):
            _, es = from_hf_entities_example({"text": text, "entities": [e]}, label_map)
            out.extend(es)
    return text, out


CAIL_SPAN_INCLUSIVE = False

def from_cail_json(obj, label_map):
    text = obj.get("context") or obj.get("text") or obj.get("sentence") or ""
    out = []
    for ent in obj.get("entities", []):
        if not isinstance(ent, dict):
            continue
        lab = _map_label(ent.get("label"), label_map)
        for sp in ent.get("span", []):
            try:
                a, b = str(sp).split(";"); s, e = int(a), int(b)
            except Exception:
                continue
            end = e + 1 if CAIL_SPAN_INCLUSIVE else e
            if 0 <= s < end <= len(text):
                out.append([s, end, lab])
    return text, out


def _parse_record(ex, label_map):
    if (isinstance(ex, dict) and isinstance(ex.get("entities"), list) and ex["entities"]
            and isinstance(ex["entities"][0], dict) and "span" in ex["entities"][0]):
        return from_cail_json(ex, label_map)
    text, ents = from_hf_entities_example(ex, label_map)
    if not ents and isinstance(ex, dict) and ("label" in ex or "spans" in ex):
        text, ents = from_jsonl_spans(ex, label_map)
    return text, ents


def try_github_raw(urls, raw_dir):
    import urllib.request
    os.makedirs(raw_dir, exist_ok=True)
    got = 0
    for url in urls:
        dst = os.path.join(raw_dir, os.path.basename(url))
        if os.path.exists(dst):
            got += 1; continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "OntoSpan-Research/1.0 (academic)"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            open(dst, "wb").write(data)
            print(f"  - downloaded {url} -> {dst} ({len(data)} bytes)"); got += 1
        except Exception as e:
            print(f"  - skipped {url}: {type(e).__name__}: {str(e)[:70]}")
    return got > 0


def try_load_hf(hf_id):
    try:
        from datasets import load_dataset
    except Exception:
        print("  `datasets` not installed; pip install datasets"); return None
    try:
        print(f"  - trying HuggingFace: {hf_id} ...")
        ds = load_dataset(hf_id)
        return {sp: list(ds[sp]) for sp in ds.keys()}
    except Exception as e:
        print(f"  HF load failed ({hf_id}): {type(e).__name__}: {str(e)[:120]}"); return None


def read_raw_dir(raw_dir, label_map):
    found = {}
    for path in sorted(glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True)):
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path).lower()
        split = "train" if "train" in name else "dev" if ("dev" in name or "val" in name) else "test" if "test" in name else "train"
        exs = []
        try:
            if path.endswith((".json", ".jsonl")):
                raw = open(path, encoding="utf-8").read().strip()
                try:
                    data = json.loads(raw)
                    items = data if isinstance(data, list) else data.get("data", [data])
                except json.JSONDecodeError:
                    items = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]
                for ex in items:
                    text, ents = _parse_record(ex, label_map)
                    if text:
                        exs.append((text, ents))
            elif path.endswith((".conll", ".txt", ".bio")):
                exs = from_bio_lines(open(path, encoding="utf-8").readlines(), label_map)
        except Exception as e:
            print(f"  ! skipped {path}: {type(e).__name__}: {str(e)[:80]}"); continue
        if exs:
            found.setdefault(split, []).extend(exs)
            print(f"  - read {path} -> {len(exs)} sentences (split {split})")
    return found


def normalize_hf(splits_examples, label_map):
    out = {}
    for sp, exs in splits_examples.items():
        norm = []
        for ex in exs:
            text, ents = from_hf_entities_example(ex, label_map)
            if not ents and ("label" in ex or "spans" in ex):
                text, ents = from_jsonl_spans(ex, label_map)
            if text:
                norm.append((text, ents))
        out[sp] = norm
    return out


def write_unified(domain, splits):
    import random
    random.seed(42)
    has = lambda k: k in splits and splits[k]
    if has("train") and has("dev") and has("test"):
        final = {k: splits[k] for k in ("train", "dev", "test")}
    elif has("train") and has("test"):
        tr = list(splits["train"]); random.shuffle(tr); cut = max(1, len(tr) // 10)
        final = {"train": tr[cut:], "dev": tr[:cut], "test": splits["test"]}
    else:
        allp = [p for sp in splits.values() for p in sp]; random.shuffle(allp)
        n = len(allp)
        final = {"train": allp[n // 5:], "dev": allp[n // 10:n // 5], "test": allp[:n // 10]}
    outdir = os.path.join(OUT_ROOT, domain); os.makedirs(outdir, exist_ok=True)
    counts = {}
    for sp, pairs in final.items():
        path = os.path.join(outdir, f"{sp}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i, (text, ents) in enumerate(pairs):
                ents = [[s, e, l] for s, e, l in ents if 0 <= s < e <= len(text)]
                f.write(json.dumps({"id": f"{domain}_{sp}_{i:06d}", "text": text, "entities": ents}, ensure_ascii=False) + "\n")
        counts[sp] = len(pairs)
    n_ent = sum(len(e) for pairs in final.values() for _, e in pairs)
    labset = sorted({l for pairs in final.values() for _, e in pairs for _, _, l in e})
    print(f"  wrote data/{domain}/  splits={counts}  entities={n_ent}  types({len(labset)})={labset}")


def handle(name, hf_id=None, raw_dir=None):
    cfg = SOURCES[name]; domain, label_map = cfg["domain"], cfg["label_map"]
    print(f"\n=== {name}  →  data/{domain}/ ===")
    raw_dir = raw_dir or os.path.join(RAW_ROOT, name)
    has_raw = os.path.isdir(raw_dir) and glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True)
    if cfg.get("github_raw") and not has_raw:
        print(f"  - fetching {name} from GitHub raw ...")
        try_github_raw(cfg["github_raw"], raw_dir)
    if os.path.isdir(raw_dir) and glob.glob(os.path.join(raw_dir, "**", "*"), recursive=True):
        got = read_raw_dir(raw_dir, label_map)
        if got:
            write_unified(domain, got); return True
    candidates = ([hf_id] if hf_id else []) + cfg["hf_candidates"]
    for cid in candidates:
        loaded = try_load_hf(cid)
        if loaded:
            write_unified(domain, normalize_hf(loaded, label_map)); return True
    print(f"  could not auto-fetch {name}.")
    print(f"    official source: {cfg['official']}")
    print(f"    manual: {cfg['manual']}")
    print(f"    expected raw dir: {raw_dir}/")
    return False


def check():
    print("=== data readiness check ===")
    for name, cfg in SOURCES.items():
        d = cfg["domain"]; outdir = os.path.join(OUT_ROOT, d)
        files = {sp: os.path.join(outdir, f"{sp}.jsonl") for sp in ("train", "dev", "test")}
        have = {sp: (os.path.exists(p) and sum(1 for _ in open(p, encoding="utf-8"))) for sp, p in files.items()}
        ready = all(isinstance(v, int) and v > 0 for v in have.values())
        raw = os.path.join(RAW_ROOT, name)
        raw_n = len(glob.glob(os.path.join(raw, "**", "*"), recursive=True)) if os.path.isdir(raw) else 0
        print(f"  {name:7s}({d:7s})  unified: {'ok '+str(have) if ready else 'not-ready '+str(have)}   raw/{name}: {raw_n} files")


def main():
    ap = argparse.ArgumentParser(description="OntoSpan: download and normalize public benchmarks")
    ap.add_argument("--dataset", choices=list(SOURCES) + ["all"], help="dataset to process")
    ap.add_argument("--hf-id", help="override/specify the HuggingFace dataset id")
    ap.add_argument("--raw-dir", help="raw dir for manually downloaded files (default data/raw/<dataset>/)")
    ap.add_argument("--check", action="store_true", help="check only, no download")
    a = ap.parse_args()
    if a.check or not a.dataset:
        check()
        if not a.dataset:
            print("\nspecify --dataset {cmeee|cail|tcmner|all} to download/normalize.")
        return
    names = list(SOURCES) if a.dataset == "all" else [a.dataset]
    ok = {n: handle(n, hf_id=a.hf_id if len(names) == 1 else None, raw_dir=a.raw_dir if len(names) == 1 else None) for n in names}
    print("\n=== results ===")
    for n, v in ok.items():
        print(f"  {n}: {'ready' if v else 'manual (see steps above)'}")
    print("\nnext: python ontospan_experiment.py --domain medical --ontology implicit   (run --check first)")


if __name__ == "__main__":
    main()
