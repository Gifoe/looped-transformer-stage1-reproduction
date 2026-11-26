"""Try adding 2-3 extra diverse content-varying unseen pools to UNSEEN_POOLS_MANIFEST.json."""
import json
from pathlib import Path
import sys
sys.path.insert(0, "/root/autodl-tmp/ouro-v4-cfit-20261126")
from v4_wave2_prepare_unseen6 import (
    OUT_DIR, build_forbidden_index, clean_and_save, fetch_hf_parquet_rows
)

def fetch_extra(name):
    recs = []
    if name == "abductive_nli":
        rows = fetch_hf_parquet_rows("allenai/art", None, "train", max_files=1)
        for i, x in enumerate(rows[:10000]):
            q = f"Beginning: {x['observation_1']}\nEnding: {x['observation_2']}\nWhat happened in between?"
            opts = [x["hypothesis_1"], x["hypothesis_2"]]
            g = int(x["label"]) - 1
            recs.append({"sample_id": f"art/train/{i}", "question": q, "options": opts, "gold": g})
    elif name == "medqa_usmle":
        rows = fetch_hf_parquet_rows("GBaker/MedQA-USMLE-4-options", None, "train", max_files=1)
        for i, x in enumerate(rows):
            od = x["options"]
            keys = sorted(od.keys())
            opts = [od[k] for k in keys]
            ans = x["answer_idx"]
            if ans not in keys:
                continue
            recs.append({"sample_id": f"medqa/train/{i}", "question": x["question"], "options": opts, "gold": keys.index(ans)})
    elif name == "reclor":
        rows = fetch_hf_parquet_rows("metaeval/reclor", None, "train", max_files=1)
        for i, x in enumerate(rows):
            q = f"{x['context']}\nQuestion: {x['question']}"
            opts = list(x["answers"])
            recs.append({"sample_id": f"reclor/train/{i}", "question": q, "options": opts, "gold": int(x["label"])})
    elif name == "ag_news":
        rows = fetch_hf_parquet_rows("fancyzhx/ag_news", None, "train", max_files=1)
        opts = ["World", "Sports", "Business", "Science and Technology"]
        for i, x in enumerate(rows[:8000]):
            q = f"Article: {x['text']}\nCategory:"
            recs.append({"sample_id": f"ag_news/train/{i}", "question": q, "options": opts, "gold": int(x["label"])})
    elif name == "hellaswag_noop":
        pass
    elif name == "math_qa":
        rows = fetch_hf_parquet_rows("allenai/math_qa", None, "train", max_files=1)
        import re
        for i, x in enumerate(rows[:10000]):
            raw_opts = re.split(r"\s*,\s*(?=[a-e]\s*\))", x["options"].strip())
            if len(raw_opts) < 4:
                continue
            opts = [re.sub(r"^[a-e]\s*\)\s*", "", o).strip() for o in raw_opts[:5]]
            c = str(x["correct"]).strip().lower()
            if not c or c[0] not in "abcde":
                continue
            g = ord(c[0]) - ord("a")
            if g < len(opts):
                recs.append({"sample_id": f"math_qa/train/{i}", "question": x["Problem"], "options": opts, "gold": g})
    return recs

def main():
    f_exact, f_grams = build_forbidden_index()
    mpath = OUT_DIR.parent / "UNSEEN_POOLS_MANIFEST.json"
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    for name in ["abductive_nli", "medqa_usmle", "reclor", "ag_news", "math_qa"]:
        if name in manifest:
            continue
        try:
            recs = fetch_extra(name)
            info = clean_and_save(name, recs, f_exact, f_grams, max_keep=6000)
            if info["n"] >= 1000:
                manifest[name] = info
        except Exception as e:
            print(f"  SKIP {name}: {type(e).__name__}: {e}", flush=True)
    mpath.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nTOTAL UNSEEN POOLS ({len(manifest)}): {list(manifest.keys())}", flush=True)

if __name__ == "__main__":
    main()
