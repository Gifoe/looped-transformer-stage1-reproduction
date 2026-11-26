"""Prepare 6-8 brand-new, 100% unseen external prospective training sources (CPU-only, pure urllib + pyarrow.parquet via hf-mirror.com).
Strictly deduplicated (exact normalized + 8-gram) against UNIFIED_EXT_DEV and all 14 existing pools!
"""

import hashlib
import io
import json
import os
import re
import unicodedata
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

HF_BASE = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
OUT_DIR = Path("/root/autodl-tmp/ouro-v4-cfit-20261126/prospective_unseen6/pools")
OUT_DIR.mkdir(parents=True, exist_ok=True)
V3_ROOT = Path("/root/ouro-v3-controllability-20261126")


def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def ngrams8(s: str):
    toks = norm_text(s).split()
    if len(toks) < 8:
        return set()
    return {" ".join(toks[i:i + 8]) for i in range(len(toks) - 7)}


def http_get_bytes(url: str, timeout=45) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ouro-v4-cfit/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_hf_parquet_rows(repo: str, config: str | None = None, split: str = "train", max_files: int = 2):
    """Query hf-mirror parquet API and read rows via pyarrow.parquet without needing `datasets`."""
    api_url = f"{HF_BASE}/api/datasets/{repo}/parquet"
    meta = json.loads(http_get_bytes(api_url, timeout=30).decode("utf-8"))
    if isinstance(meta, dict):
        if config is None:
            config = "default" if "default" in meta else next(iter(meta.keys()))
        cfg_map = meta[config]
        urls = cfg_map.get(split)
        if not urls:
            # try matching prefix e.g. train_r1
            for k, v in cfg_map.items():
                if k == split or k.startswith(split):
                    urls = v
                    break
        if not urls:
            raise KeyError(f"Split {split} not found in {repo}/{config}: {list(cfg_map.keys())}")
    else:
        raise ValueError(f"Unexpected parquet metadata for {repo}: {type(meta)}")

    rows = []
    for u in urls[:max_files]:
        # Rewrite huggingface.co to HF_BASE if API returns canonical URLs
        u2 = u.replace("https://huggingface.co", HF_BASE)
        data = http_get_bytes(u2, timeout=60)
        tbl = pq.read_table(io.BytesIO(data))
        rows.extend(tbl.to_pylist())
    return rows


def build_forbidden_index():
    """Load all UNIFIED_EXT_DEV texts so zero overlap can ever leak in."""
    exact = set()
    grams = set()
    dev_dir = V3_ROOT / "stage0_env" / "UNIFIED_EXT_DEV"
    for p in dev_dir.glob("*.jsonl"):
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            for k in ("question", "stem", "context", "query"):
                if isinstance(r.get(k), str) and len(r[k]) > 10:
                    exact.add(norm_text(r[k]))
                    grams.update(ngrams8(r[k]))
    print(f"Forbidden dev index: {len(exact)} exact strings, {len(grams)} 8-grams", flush=True)
    return exact, grams


def clean_and_save(name, raw_records, forbidden_exact, forbidden_grams, max_keep=6000):
    kept = []
    excluded = 0
    seen_q = set()
    for r in raw_records:
        q = str(r["question"]).strip()
        opts = [str(o).strip() for o in r["options"]]
        g = int(r["gold"])
        if not q or len(opts) < 2 or len(opts) > 8 or not (0 <= g < len(opts)):
            continue
        if any(not o for o in opts) or len(set(norm_text(o) for o in opts)) != len(opts):
            continue
        nq = norm_text(q)
        if nq in seen_q or nq in forbidden_exact:
            excluded += 1
            continue
        if ngrams8(q) & forbidden_grams:
            excluded += 1
            continue
        seen_q.add(nq)
        labels = [chr(65 + i) for i in range(len(opts))]
        sid = str(r["sample_id"])
        kept.append({
            "sample_id": sid,
            "dataset": name,
            "task_type": "mc_content",
            "options": opts,
            "label": g,
            "answer_content": opts[g],
            "candidate": {
                "id": sid,
                "question": q,
                "choices": {"label": labels, "text": opts},
                "answerKey": labels[g],
            },
        })
    kept.sort(key=lambda x: hashlib.sha256(f"v4_unseen:{x['sample_id']}".encode()).hexdigest())
    kept = kept[:max_keep]
    out_path = OUT_DIR / f"{name}.jsonl"
    out_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in kept), encoding="utf-8")
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"  SAVED {name:14s}: n={len(kept):5d} excluded={excluded:3d} sha={sha[:16]}...", flush=True)
    return {"name": name, "path": str(out_path), "n": len(kept), "excluded": excluded, "sha256": sha}


def fetch_source(name):
    recs = []
    if name == "quartz":
        rows = fetch_hf_parquet_rows("allenai/quartz", None, "train")
        for i, x in enumerate(rows):
            q = (str(x["para"]).strip() + "\n" + str(x["question"]).strip()).strip()
            opts = list(x["choices"]["text"])
            lbls = list(x["choices"]["label"])
            if x["answerKey"] not in lbls:
                continue
            g = lbls.index(x["answerKey"])
            recs.append({"sample_id": f"quartz/train/{x.get('id', i)}", "question": q, "options": opts, "gold": g})
    elif name == "dream":
        rows = fetch_hf_parquet_rows("dataset-org/dream", None, "train")
        for i, x in enumerate(rows):
            dial = "\n".join(x["dialogue"])
            q = f"{dial}\nQuestion: {x['question']}"
            opts = list(x["choice"])
            if x["answer"] not in opts:
                continue
            g = opts.index(x["answer"])
            recs.append({"sample_id": f"dream/train/{x.get('id', i)}", "question": q, "options": opts, "gold": g})
    elif name == "swag":
        rows = fetch_hf_parquet_rows("allenai/swag", "regular", "train", max_files=1)
        for i, x in enumerate(rows):
            q = (str(x["sent1"]).strip() + " " + str(x["sent2"]).strip()).strip()
            opts = [x[f"ending{k}"] for k in range(4)]
            recs.append({"sample_id": f"swag/train/{i}", "question": q, "options": opts, "gold": int(x["label"])})
    elif name == "aqua_rat":
        rows = fetch_hf_parquet_rows("deepmind/aqua_rat", "raw", "train", max_files=1)
        for i, x in enumerate(rows):
            opts = [re.sub(r"^[A-E]\)\s*", "", str(o)).strip() for o in x["options"]]
            corr = str(x["correct"]).strip()
            if not corr or corr[0] not in "ABCDE":
                continue
            g = ord(corr[0]) - 65
            recs.append({"sample_id": f"aqua_rat/train/{i}", "question": x["question"], "options": opts, "gold": g})
    elif name == "anli":
        rows = fetch_hf_parquet_rows("facebook/anli", None, "train_r1", max_files=1)
        opts = ["entailment", "neutral", "contradiction"]
        for i, x in enumerate(rows):
            q = f"Premise: {x['premise']}\nHypothesis: {x['hypothesis']}\nThe relationship between the premise and hypothesis is"
            recs.append({"sample_id": f"anli_r1/train/{x.get('uid', i)}", "question": q, "options": opts, "gold": int(x["label"])})
    elif name == "scitail":
        rows = fetch_hf_parquet_rows("allenai/scitail", "snli_format", "train", max_files=1)
        opts = ["entails", "neutral"]
        lmap = {"entailment": 0, "neutral": 1}
        for i, x in enumerate(rows):
            if x.get("gold_label") not in lmap:
                continue
            q = f"Premise: {x['sentence1']}\nHypothesis: {x['sentence2']}\nThe premise"
            recs.append({"sample_id": f"scitail/train/{i}", "question": q, "options": opts, "gold": lmap[x["gold_label"]]})
    elif name == "quarel":
        rows = fetch_hf_parquet_rows("allenai/quarel", None, "train")
        for i, x in enumerate(rows):
            m = re.search(r"\(A\)\s*(.+?)\s*\(B\)\s*(.+)$", str(x["question"]), re.S)
            if not m:
                continue
            stem = str(x["question"])[:m.start()].strip()
            opts = [m.group(1).strip(), m.group(2).strip()]
            recs.append({"sample_id": f"quarel/train/{x.get('id', i)}", "question": stem, "options": opts, "gold": int(x["answer_index"])})
    elif name == "riddle_sense":
        rows = fetch_hf_parquet_rows("INK-USC/riddle_sense", None, "train")
        for i, x in enumerate(rows):
            opts = list(x["choices"]["text"])
            lbls = list(x["choices"]["label"])
            if x["answerKey"] not in lbls:
                continue
            g = lbls.index(x["answerKey"])
            recs.append({"sample_id": f"riddle_sense/train/{i}", "question": x["question"], "options": opts, "gold": g})
    elif name == "logiqa":
        rows = fetch_hf_parquet_rows("lucasmccabe/logiqa", None, "train")
        for i, x in enumerate(rows):
            q = f"{x['context']}\nQuestion: {x['query']}"
            opts = list(x["options"])
            recs.append({"sample_id": f"logiqa/train/{i}", "question": q, "options": opts, "gold": int(x["correct_option"])})
    elif name == "mnli":
        rows = fetch_hf_parquet_rows("nyu-mll/multi_nli", None, "train", max_files=1)
        opts = ["entailment", "neutral", "contradiction"]
        for i, x in enumerate(rows[:10000]):
            q = f"Premise: {x['premise']}\nHypothesis: {x['hypothesis']}\nThe relationship is"
            recs.append({"sample_id": f"mnli/train/{i}", "question": q, "options": opts, "gold": int(x["label"])})
    return recs


def main():
    f_exact, f_grams = build_forbidden_index()
    candidates = ["quartz", "quarel", "dream", "swag", "riddle_sense", "logiqa", "aqua_rat", "anli", "scitail", "mnli"]
    manifest = {}
    for name in candidates:
        try:
            recs = fetch_source(name)
            info = clean_and_save(name, recs, f_exact, f_grams, max_keep=6000)
            if info["n"] >= 1000:
                manifest[name] = info
        except Exception as e:
            print(f"  SKIP {name}: {type(e).__name__}: {e}", flush=True)
    mpath = OUT_DIR.parent / "UNSEEN_POOLS_MANIFEST.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nPREPARED {len(manifest)} UNSEEN PROSPECTIVE POOLS: {list(manifest.keys())}", flush=True)


if __name__ == "__main__":
    main()
