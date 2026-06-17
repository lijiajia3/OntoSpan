#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Few-shot LLM baseline via the DeepSeek API.
#
# Uses the same test set and span-level metric as the trained baselines; prompts
# the model to extract typed entities as JSON. Requires DEEPSEEK_API_KEY.
import os, sys, json, time, argparse, re

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "run_code"))


def load_test_train(domain, test_cap=300, train_n=200):
    from run_code.run_grid_safe import _subsample
    from run_code.ontospan_experiment import read_jsonl, build_types, load_ontology
    ddir = os.path.join(ROOT, "data", domain)
    paths = {sp: os.path.join(ddir, f"{sp}.jsonl") for sp in ("train", "dev", "test")}
    miss = [p for p in paths.values() if not os.path.exists(p)]
    if miss:
        print(f"missing data: {miss}\n  tcm: `python run_code/download_datasets.py --dataset tcmner`;"
              f"medical (CMeEE) is licence-restricted; copy data/medical/ from where you have it.")
        sys.exit(1)
    train = read_jsonl(paths["train"]); dev = read_jsonl(paths["dev"]); test = read_jsonl(paths["test"])
    test = _subsample(test, test_cap, seed=42)
    onto = load_ontology(domain); types = build_types(train + dev, onto)
    return train[:train_n], test, [t for t in types if t != "∅"]


def build_fewshot(train, types, shots):
    ex = []
    for text, ents in train:
        if not ents:
            continue
        items = [{"text": text[cs:ce], "type": lab} for cs, ce, lab in ents]
        ex.append((text, items))
        if len(ex) >= shots:
            break
    return ex


SYS_TMPL = (
    "你是中文命名实体识别标注器。只抽取下列类型的实体,逐字照抄原文中的实体串(不要改写、不要归一)。"
    "类型集合:{types}。\n"
    "输出严格 JSON:{{\"entities\":[{{\"text\":\"实体串\",\"type\":\"类型\"}}, ...]}};没有实体则 entities 为空数组。"
    "不要输出任何解释。"
)


def make_messages(sys_prompt, fewshot, sentence):
    msgs = [{"role": "system", "content": sys_prompt}]
    for text, items in fewshot:
        msgs.append({"role": "user", "content": f"句子:{text}"})
        msgs.append({"role": "assistant", "content": json.dumps({"entities": items}, ensure_ascii=False)})
    msgs.append({"role": "user", "content": f"句子:{sentence}"})
    return msgs


def parse_entities(raw):
    if not raw:
        return []
    try:
        obj = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return []
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return []
    ents = obj.get("entities", obj) if isinstance(obj, dict) else obj
    out = []
    if isinstance(ents, list):
        for e in ents:
            if isinstance(e, dict) and e.get("text") and e.get("type"):
                out.append((str(e["text"]), str(e["type"])))
    return out


def to_spans(sentence, ents, valid_types):
    pred = set()
    for txt, typ in ents:
        if typ not in valid_types or not txt:
            continue
        start = sentence.find(txt)
        while start != -1:
            pred.add((start, start + len(txt), typ))
            start = sentence.find(txt, start + 1)
    return pred


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["tcm", "medical", "law"])
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--test-cap", type=int, default=300)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()

    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        print("DEEPSEEK_API_KEY not set; run: export DEEPSEEK_API_KEY=sk-xxxx"); sys.exit(1)
    try:
        from openai import OpenAI
    except ImportError:
        print("openai not installed: pip install openai"); sys.exit(1)
    client = OpenAI(api_key=key, base_url=a.base_url)

    train, test, types = load_test_train(a.domain, a.test_cap)
    fewshot = build_fewshot(train, types, a.shots)
    sys_prompt = SYS_TMPL.format(types="、".join(types))
    print(f"domain={a.domain} | test={len(test)} sents | types({len(types)})={types} | shots={len(fewshot)} | model={a.model}")

    cache_dir = os.path.join(ROOT, "run_code", "results_baselines", ".llm_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{a.domain}_{a.model}.jsonl")
    cache = {}
    if os.path.exists(cache_path):
        for ln in open(cache_path, encoding="utf-8"):
            try:
                d = json.loads(ln); cache[d["i"]] = d["raw"]
            except Exception:
                pass
    cf = open(cache_path, "a", encoding="utf-8")

    tp = fp = fn = 0; per = {}; t0 = time.time(); n_api = 0
    for i, (sentence, gold) in enumerate(test):
        gold_set = {(cs, ce, lab) for cs, ce, lab in gold}
        if i in cache:
            raw = cache[i]
        else:
            raw = None
            for attempt in range(4):
                try:
                    resp = client.chat.completions.create(
                        model=a.model, messages=make_messages(sys_prompt, fewshot, sentence),
                        temperature=0.0, max_tokens=a.max_tokens,
                        response_format={"type": "json_object"})
                    raw = resp.choices[0].message.content; break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"   [{i}] API error ({type(e).__name__}), retry in {wait}s...", flush=True); time.sleep(wait)
            if raw is None:
                raw = '{"entities":[]}'
            cf.write(json.dumps({"i": i, "raw": raw}, ensure_ascii=False) + "\n"); cf.flush(); n_api += 1
        pred_set = to_spans(sentence, parse_entities(raw), types)
        for k in pred_set:
            per.setdefault(k[2], [0, 0, 0])
            if k in gold_set: tp += 1; per[k[2]][0] += 1
            else: fp += 1; per[k[2]][1] += 1
        for k in gold_set:
            per.setdefault(k[2], [0, 0, 0])
            if k not in pred_set: fn += 1; per[k[2]][2] += 1
        if (i + 1) % 25 == 0:
            P, R, Fm = prf(tp, fp, fn)
            print(f"  {i+1}/{len(test)}  running micro-F1={Fm:.4f}  (API calls {n_api}, {(time.time()-t0)/60:.1f}min)", flush=True)
    cf.close()

    P, R, Fm = prf(tp, fp, fn)
    res = {"domain": a.domain, "kind": "llm_baseline", "model": a.model, "shots": len(fewshot),
           "n_test": len(test), "P": P, "R": R, "F1": Fm, "tp": tp, "fp": fp, "fn": fn,
           "per_type": {l: dict(zip(("P", "R", "F1"), prf(*v))) for l, v in sorted(per.items())}}
    out = os.path.join(ROOT, "run_code", "results_baselines", f"{a.domain}_llm_deepseek.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n" + "=" * 60)
    print(f"DeepSeek({a.model}) few-shot | domain={a.domain}")
    print(f"  micro  P={P:.4f}  R={R:.4f}  F1={Fm:.4f}   (test {len(test)} sents, {len(fewshot)}-shot)")
    print(f"  same test set and protocol as OntoSpan(explicit)/biaffine/GP/CRF.")
    print(f"  ✓ saved {out}")


if __name__ == "__main__":
    main()
