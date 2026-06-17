# -*- coding: utf-8 -*-
# Entity dictionaries and signed-ontology edges (mutex / attractive) per domain.
#
# Defines ENTITY_DICT and the signed-ontology edges (MUTEX_TYPE / MUTEX_INST /
# ATTRACT), including the pharmacopoeial eighteen-incompatibilities / nineteen-
# antagonisms herb pairs. Also holds an auxiliary silver-annotation guideline generator.

import os, re, json, time, random, argparse, datetime
random.seed(42)


_FAN = [
    (["甘草"], ["甘遂", "大戟", "海藻", "芫花"]),
    (["乌头", "川乌", "草乌", "附子"],
     ["半夏", "瓜蒌", "天花粉", "贝母", "川贝母", "浙贝母", "白蔹", "白及"]),
    (["藜芦"], ["人参", "西洋参", "党参", "沙参", "南沙参", "北沙参", "丹参",
                "玄参", "苦参", "细辛", "芍药", "赤芍", "白芍"]),
]
_WEI = [
    ("硫黄", "朴硝"), ("硫黄", "芒硝"), ("水银", "砒霜"), ("狼毒", "密陀僧"),
    ("巴豆", "牵牛"), ("丁香", "郁金"), ("芒硝", "三棱"), ("牙硝", "三棱"),
    ("川乌", "犀角"), ("草乌", "犀角"), ("乌头", "犀角"),
    ("人参", "五灵脂"), ("肉桂", "赤石脂"), ("官桂", "赤石脂"),
]
_seen, TCM_MUTEX_PAIRS = set(), []
for _heads, _xs in _FAN:
    for _h in _heads:
        for _x in _xs:
            if frozenset((_h, _x)) not in _seen:
                _seen.add(frozenset((_h, _x))); TCM_MUTEX_PAIRS.append([_h, _x])
for _a, _b in _WEI:
    if frozenset((_a, _b)) not in _seen:
        _seen.add(frozenset((_a, _b))); TCM_MUTEX_PAIRS.append([_a, _b])
TCM_HERBS = sorted({h for p in TCM_MUTEX_PAIRS for h in p} | {
    "桂枝","白芍","生姜","大枣","黄芪","当归","熟地黄","茯苓","白术","柴胡","黄芩","陈皮","厚朴","枳实","大黄","芒硝","麻黄","杏仁","石膏","知母"})

MED_MUTEX_PAIRS = [
    ["华法林","阿司匹林"],["华法林","阿托伐他汀"],["辛伐他汀","克拉霉素"],["辛伐他汀","红霉素"],
    ["螺内酯","氯化钾"],["地高辛","胺碘酮"],["甲氨蝶呤","复方新诺明"],["氟西汀","司来吉兰"],
]
MED_DRUGS = sorted({d for p in MED_MUTEX_PAIRS for d in p} | {
    "二甲双胍","胰岛素","硝苯地平","美托洛尔","氨氯地平","氯吡格雷","头孢曲松","左氧氟沙星"})

ENTITY_DICT = {
    "law": {
        "案由": ["买卖合同纠纷","借款合同纠纷","机动车交通事故责任纠纷","离婚纠纷","劳动争议","房屋租赁合同纠纷","民间借贷纠纷"],
    },
    "tcm": {
        "中药": TCM_HERBS,
        "方剂": ["桂枝汤","麻黄汤","六味地黄丸","四物汤","小柴胡汤","逍遥散","补中益气汤","金匮肾气丸"],
        "证候": ["脾胃虚弱","肝郁气滞","气血两虚","肾阳虚","痰湿内阻","气滞血瘀","阴虚火旺"],
        "疾病": ["消渴","痹证","眩晕","咳嗽","泄泻","不寐"],
        "症状": ["头痛","恶心","发热","乏力","纳差","舌苔黄腻","脉弦细","口干"],
        "炮制方法": ["酒炙","蜜炙","醋炙","盐炙","煅","炒","炮"],
        "剂型": ["汤剂","丸剂","散剂","膏剂","颗粒","胶囊"],
        "用法用量": ["水煎服","每日一剂","每日三次"],
    },
    "medical": {
        "药物": MED_DRUGS,
        "疾病": ["2型糖尿病","原发性高血压","急性心肌梗死","心房颤动","慢性肾脏病","肺炎","脑梗死"],
        "临床表现": ["发热","胸痛","出血","头晕","呼吸困难","水肿","血压升高","心悸"],
        "检验项目": ["血常规","糖化血红蛋白","凝血功能","肝功能","肌酐","血钾"],
        "科室": ["心血管内科","内分泌科","急诊科","呼吸内科","肾内科"],
        "身体部位": ["左心室","肝脏","肾脏","冠状动脉","胃"],
        "微生物": ["幽门螺杆菌","金黄色葡萄球菌","大肠杆菌","肺炎链球菌"],
        "医疗程序": ["冠状动脉造影","血液透析","机械通气","经皮冠状动脉介入治疗"],
        "医疗设备": ["呼吸机","起搏器","除颤仪"],
    },
}

LAW_REGEX = [
    ("法条", re.compile(r"《[^》]{2,30}》(?:第[一二三四五六七八九十百千零\d]+条(?:第[一二三四五六七八九十\d]+款)?)?")),
    ("罪名", re.compile(r"[一-鿿]{2,8}罪")),
    ("法院", re.compile(r"[一-鿿]{2,15}?(?:高级|中级|基层)?人民法院")),
    ("刑期", re.compile(r"(?:有期徒刑|拘役|管制)[一二三四五六七八九十百\d]{1,4}(?:年|个月|月)(?:[一二三四五六七八九十百\d]{1,3}(?:个月|月))?|无期徒刑")),
]
LAW_ROLE_RE = re.compile(r"(原告|被告|上诉人|被上诉人|第三人)[:：]?([一-鿿]{2,12}?(?:公司|银行|医院|厂|某|某某)?)(?=诉|与|不服|因|向|，|。|；|、|提起|答辩|$)")

MUTEX_TYPE = {"law": [["原告","被告"],["上诉人","被上诉人"],["犯罪嫌疑人","受害人"]], "tcm": [], "medical": []}
MUTEX_INST = {"law": [], "tcm": TCM_MUTEX_PAIRS, "medical": MED_MUTEX_PAIRS}
ATTRACT = {
    "law": [["原告","第三人","related-to"],["罪名","法条","related-to"]],
    "tcm": [["症状","证候","related-to"],["中药","方剂","related-to"]],
    "medical": [["临床表现","疾病","related-to"],["药物","疾病","related-to"]],
}

TYPE_GUIDE = {
    "law": {
        "原告":("本案提起诉讼的当事人名称","原告**李某**诉…→标“李某”","不含“原告”二字；与被告互斥"),
        "被告":("本案被起诉的当事人名称","被告**某科技公司**","与原告互斥；反诉中以本句语境为准"),
        "第三人":("与案件结果有利害关系、参加诉讼的当事人","第三人**王某**","区别于原/被告"),
        "上诉人":("二审中提起上诉的当事人","上诉人**张某**","与被上诉人互斥"),
        "被上诉人":("二审中被上诉的当事人","被上诉人**某公司**","与上诉人互斥"),
        "法院":("审理或作出裁判的法院","**北京市第一中级人民法院**","取完整层级名称"),
        "案由":("案件性质的规范名称","**买卖合同纠纷**","取规范案由全称"),
        "罪名":("刑事案件的具体罪名","**盗窃罪**","含“罪”字；一案多罪分别标"),
        "法条":("引用的法律条文","《民法典》**第一千一百六十五条**","法规名+条款号，可嵌套"),
        "刑期":("判处的刑罚期限","**有期徒刑三年**","含刑种+期限"),
    },
    "tcm": {
        "中药":("单味中药材名称","**乌头**、**半夏**","标本名，不含剂量；炮制前缀并入"),
        "方剂":("复方制剂/方名","**桂枝汤**","含汤/丸/散等后缀；别与单味中药混"),
        "证候":("中医辨证的证型","**肝郁气滞**","辨证术语，别与疾病/症状混"),
        "疾病":("疾病名称","**消渴**、**痹证**","区别于证候与症状"),
        "症状":("症状或体征","**头痛**、**舌苔黄腻**","单个症状为一跨度"),
        "炮制方法":("中药炮制工艺","**酒炙**、**煅**","工艺动词"),
        "剂型":("药物形态","**汤剂**、**丸剂**","别与方名里的“汤/丸”混"),
        "用法用量":("服用方法与剂量","**每日三次**、**水煎服**","含频次/剂量/给药方式"),
        "古籍出处":("方剂/论述出自的典籍","《**伤寒论**》","取书名号内"),
    },
    "medical": {
        "疾病":("疾病/诊断名称","**2型糖尿病**","含分型/分期"),
        "临床表现":("症状与体征","**胸痛**、**血压升高**","别与检验结果混"),
        "药物":("药品/药物名称","**华法林**、**他汀类**","标药名不含剂量；配伍禁忌即建于此类实例间"),
        "医疗程序":("诊疗操作/手术","**冠状动脉造影**","操作名"),
        "医疗设备":("器械/设备","**呼吸机**","实体器械"),
        "身体部位":("解剖部位","**左心室**","解剖名词"),
        "检验项目":("检查/化验项目","**糖化血红蛋白**","项目名，区别于其结果"),
        "科室":("临床科室","**心血管内科**","科室名"),
        "微生物":("病原微生物","**幽门螺杆菌**","病原体名"),
    },
}

TARGET = {"law": 1500, "tcm": 1500, "medical": 2500}
MUTEX_QUOTA = {"law": 0.5, "tcm": 0.6, "medical": 0.5}

SEEDS = {
    "law": ["https://zh.wikisource.org/wiki/中华人民共和国民事诉讼法",
            "https://zh.wikisource.org/wiki/中华人民共和国刑法"],
    "tcm": ["https://zh.wikisource.org/wiki/傷寒論", "https://zh.wikisource.org/wiki/金匱要略"],
    "medical": ["https://zh.wikipedia.org/wiki/华法林", "https://zh.wikipedia.org/wiki/阿司匹林"],
}
OFFLINE = {
    "law": ["原告张某诉被告李某买卖合同纠纷一案，本院依法组成合议庭公开开庭审理。",
            "上诉人王某不服一审判决向本院提起上诉，被上诉人某科技公司答辩称其不构成违约。",
            "被告人因犯盗窃罪，被判处有期徒刑三年，并处罚金。",
            "本院认为，依据《中华人民共和国民法典》第五百七十七条，被告应承担违约责任。"],
    "tcm": ["方中乌头与半夏同用属十八反之配伍禁忌，临床当慎用。",
            "甘草反甘遂、大戟、海藻、芫花，组方时须避之。",
            "人参畏五灵脂，二者不宜同方；本方以桂枝汤加减治太阳中风。",
            "患者证属肝郁气滞，症见头痛、纳差、脉弦，治以逍遥散水煎服，每日一剂。"],
    "medical": ["患者长期服用华法林，加用阿司匹林后出血风险显著升高，属配伍禁忌。",
                "辛伐他汀与克拉霉素合用可增加肌病风险，应避免联用。",
                "螺内酯与氯化钾联用易致高钾血症，需监测血钾。",
                "地高辛与胺碘酮合用可升高地高辛血药浓度，宜减量并监测。"],
}

SENT_SPLIT = re.compile(r"[。！？；!?;]\s*|\n+")
SPACES = re.compile(r"[ \t　]+")
MIN_LEN, MAX_LEN = 8, 128


def split_sentences(text):
    for raw in SENT_SPLIT.split(text or ""):
        s = SPACES.sub("", raw).strip()
        if MIN_LEN <= len(s) <= MAX_LEN and re.search(r"[一-鿿]", s):
            yield s


def fetch_online(domain, limit):
    try:
        import requests
    except ImportError:
        print("  --online needs requests: pip install requests beautifulsoup4; skipping web fetch."); return []
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        BeautifulSoup = None
    sents, ua = [], {"User-Agent": "OntoSpan-Research/1.0 (academic)"}
    for url in SEEDS[domain]:
        try:
            html = requests.get(url, headers=ua, timeout=20).text
            text = (BeautifulSoup(html, "html.parser").get_text("\n") if BeautifulSoup
                    else re.sub(r"<[^>]+>", "\n", html))
            ps = list(split_sentences(text)); sents += ps
            print(f"  fetched {url} -> {len(ps)} sentences"); time.sleep(1.5)
        except Exception as e:
            print(f"  fetch failed {url}: {e}")
        if len(sents) >= limit: break
    return sents


def gather_sentences(domain, online, limit):
    sents = list(OFFLINE[domain])
    rawdir = os.path.join("raw", domain)
    if os.path.isdir(rawdir):
        for fn in os.listdir(rawdir):
            if fn.endswith(".txt"):
                sents += list(split_sentences(open(os.path.join(rawdir, fn), encoding="utf-8", errors="ignore").read()))
    if online:
        sents += fetch_online(domain, limit)
    seen, uniq = set(), []
    for s in sents:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq


def mutex_cues(domain):
    if domain == "law": return ["原告","被告","上诉人","被上诉人"]
    if domain == "tcm": return list({h for p in TCM_MUTEX_PAIRS for h in p}) + ["反","畏","配伍禁忌"]
    return list({d for p in MED_MUTEX_PAIRS for d in p}) + ["配伍禁忌","禁止合用","相互作用","禁忌"]


def sample(domain, sents):
    target, quota = TARGET[domain], MUTEX_QUOTA[domain]
    cues = mutex_cues(domain)
    def score(s):
        hits = sum(1 for c in cues if c in s); return hits + (5 if hits >= 2 else 0)
    scored = sorted(sents, key=score, reverse=True)
    mutex_part = [s for s in scored if score(s) >= 1][:int(target*quota)]
    rest = [s for s in sents if s not in set(mutex_part)]; random.shuffle(rest)
    picked = (mutex_part + rest)[:target]; random.shuffle(picked)
    return picked


def distant_label(domain, text):
    spans = []
    for typ, words in ENTITY_DICT.get(domain, {}).items():
        for w in sorted(words, key=len, reverse=True):
            start = 0
            while (idx := text.find(w, start)) != -1:
                spans.append([idx, idx+len(w), typ]); start = idx + len(w)
    if domain == "law":
        for m in LAW_ROLE_RE.finditer(text):
            name = m.group(2)
            if name:
                s = m.start(2); spans.append([s, s+len(name), m.group(1)])
        for typ, rgx in LAW_REGEX:
            for m in rgx.finditer(text):
                spans.append([m.start(), m.end(), typ])
    if domain in ("tcm", "medical"):
        for m in re.finditer(r"《[^》]{2,20}》", text):
            spans.append([m.start(), m.end(), "古籍出处" if domain == "tcm" else "疾病"]) if domain=="tcm" else None
    spans.sort(key=lambda x: (x[0], -(x[1]-x[0])))
    out, occupied = [], [False]*len(text)
    for s, e, t in spans:
        if not any(occupied[s:e]):
            out.append([s, e, t])
            for k in range(s, e): occupied[k] = True
    return sorted(out)


def build_ontology(domain):
    return {"domain": domain,
            "entity_types": list(TYPE_GUIDE[domain].keys()),
            "type_layer": {"attractive": ATTRACT[domain], "mutex": MUTEX_TYPE[domain]},
            "instance_layer": {"mutex_pairs": MUTEX_INST[domain]},
            "note": "type_layer.mutex 为类型级互斥(法律)；instance_layer.mutex_pairs 为实例级互斥(中医十八反十九畏/医疗配伍禁忌)。"}


def write_guideline(domain, stats):
    g = TYPE_GUIDE[domain]; L = [f"# {domain} NER annotation guideline (OntoSpan)\n",
        f"- target {TARGET[domain]} sentences; this batch {stats['n']}, with mutex cues {stats['mutex']} ({stats['mutex']/max(1,stats['n']):.0%})",
        f"- **Silver-label warning**: distantly-supervised auto pre-annotation; needs human review, Kappa>=0.85 to be gold.\n",
        "## What to label: entity spans + types only; mutex relations are supplied by the ontology (ontology.json), not labeled per sentence.\n",
        f"## Entity types ({len(g)})"]
    for t,(d,eg,b) in g.items():
        L.append(f"- **{t}**: {d}. e.g. {eg}. boundary: {b}.")
    L += ["\n## Process: (1) two annotators label the first 200 (calibration) -> Cohen's Kappa; (2) >=0.85 to proceed; (3) double-annotate all; (4) third-party adjudication; (5) 5% re-check."]
    return "\n".join(L)


def write_readme(domain, stats):
    return "\n".join([f"# {domain}-NER dataset (OntoSpan)\n",
        f"- size: {stats['n']} sentences (train/dev/test = {stats['tr']}/{stats['dv']}/{stats['te']}, 8:1:1)",
        f"- sentences with mutex cues: {stats['mutex']} ({stats['mutex']/max(1,stats['n']):.0%})",
        f"- entity types: {len(TYPE_GUIDE[domain])}; ontology: see ontology.json (type/instance-level mutex)",
        "\n## Note on labels",
        "本仓库标签为**银标(silver, 远程监督自动生成)**，非金标。直接用于训练/发表前**必须人工校验**",
        "(doccano double annotation + Cohen's Kappa>=0.85 + third-party adjudication; see guideline.md).",
        "\n## 来源与许可", "- 见 provenance.jsonl。维基文库法规/中医经典为公版；维基百科为 CC-BY-SA(需署名)；",
        "  Drug contraindications are illustrative; verify against the latest package inserts / clinical guidance before release."])


def export(domain, picked):
    out = os.path.join("datasets", domain); os.makedirs(out, exist_ok=True)
    records = [{"id": f"{domain}_{i:05d}", "text": s, "label": distant_label(domain, s),
                "batch": "calibration" if i < 200 else "main"} for i, s in enumerate(picked)]
    cues = mutex_cues(domain); n = len(records)
    n_mutex = sum(1 for r in records if sum(1 for c in cues if c in r["text"]) >= 1)
    idx = list(range(n)); random.shuffle(idx)
    te = set(idx[:n//10]); dv = set(idx[n//10:n//5])
    splits = {"train": [], "dev": [], "test": []}
    for i, r in enumerate(records):
        splits["test" if i in te else "dev" if i in dv else "train"].append(r)
    def dump(name, recs):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            for r in recs: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    dump(f"{domain}_silver.jsonl", records)
    for k, v in splits.items(): dump(f"{k}.jsonl", v)
    json.dump(build_ontology(domain), open(os.path.join(out,"ontology.json"),"w",encoding="utf-8"),
              ensure_ascii=False, indent=2)
    stats = {"n": n, "mutex": n_mutex, "tr": len(splits["train"]), "dv": len(splits["dev"]), "te": len(splits["test"])}
    open(os.path.join(out,"guideline.md"),"w",encoding="utf-8").write(write_guideline(domain, stats))
    open(os.path.join(out,"README.md"),"w",encoding="utf-8").write(write_readme(domain, stats))
    json.dump({"domain":domain,"n":n,"sources":"OFFLINE bundled + raw/ + (--online: "+";".join(SEEDS[domain])+")",
               "time":datetime.datetime.now().isoformat(timespec="seconds")},
              open(os.path.join(out,"provenance.jsonl"),"w",encoding="utf-8"), ensure_ascii=False)
    n_spans = sum(len(r["label"]) for r in records)
    print(f"[{domain}] {n} sentences (mutex cues {n_mutex}/{n_mutex/max(1,n):.0%}) | silver entities {n_spans} |"
          f"train/dev/test={stats['tr']}/{stats['dv']}/{stats['te']} → datasets/{domain}/")


def run(domain, online, limit):
    sents = gather_sentences(domain, online, limit)
    if not sents:
        print(f"[{domain}] no corpus"); return
    export(domain, sample(domain, sents))


def main():
    ap = argparse.ArgumentParser(description="OntoSpan: generate the three-domain datasets")
    ap.add_argument("--domain", choices=["law","tcm","medical"])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--online", action="store_true", help="additionally scrape from public sources (e.g. Wikisource)")
    ap.add_argument("--limit", type=int, default=2000, help="max sentences to scrape per domain")
    a = ap.parse_args()
    doms = ["law","tcm","medical"] if a.all or not a.domain else [a.domain]
    print("generated SILVER labels; require human verification + Kappa>=0.85 to be gold (see each domain README.md)\n")
    for d in doms: run(d, a.online, a.limit)
    print("\nnext: import datasets/<domain>/*_silver.jsonl into doccano for review; Kappa tool in collect_for_annotation.py")


if __name__ == "__main__":
    main()
