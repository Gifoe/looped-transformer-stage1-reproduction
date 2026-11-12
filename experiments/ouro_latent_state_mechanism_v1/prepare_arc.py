"""Prepare full official ARC-Challenge splits, with gold isolated from inference inputs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from core import DEFAULT_DEMOS, demo_prefix, sha256_file, sha256_text, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demonstrations", type=Path, default=DEFAULT_DEMOS)
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--dataset-revision", default=None, help="Optional pinned HuggingFace dataset revision")
    a = p.parse_args()
    from datasets import load_dataset
    if not a.demonstrations.is_file():
        raise FileNotFoundError(f"Historical demonstration manifest absent: {a.demonstrations}")
    demos = json.loads(a.demonstrations.read_text(encoding="utf-8"))
    prefix = demo_prefix(demos)
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", revision=a.dataset_revision)
    a.out.mkdir(parents=True, exist_ok=True)
    demo_ids = set(demos["ids"])
    metadata = {"dataset": "allenai/ai2_arc", "subset": "ARC-Challenge", "dataset_revision": a.dataset_revision,
                "demonstration_file_sha256": sha256_file(a.demonstrations), "joint_prefix_sha256": sha256_text(prefix), "demonstration_ids": demos["ids"], "splits": {}}
    for split in ("train", "validation", "test"):
        input_path = a.out / f"{split}.jsonl"
        label_path = a.out / f"{split}.labels.jsonl"
        with input_path.open("w", encoding="utf-8") as inp, label_path.open("w", encoding="utf-8") as lab:
            for doc in ds[split]:
                labels = list(doc["choices"]["label"])
                texts = list(doc["choices"]["text"])
                gold = labels.index(str(doc["answerKey"]))
                rid = str(doc["id"])
                item = {"id": f"arc_challenge/{split}/{rid}", "split": split,
                        "source_id": rid, "question": doc["question"], "choices": texts,
                        "n": len(texts), "in_fewshot_demonstrations": rid in demo_ids}
                assert 2 <= len(texts) <= 8
                inp.write(json.dumps(item, ensure_ascii=False) + "\n")
                lab.write(json.dumps({"id": item["id"], "gold": gold}, ensure_ascii=False) + "\n")
        metadata["splits"][split] = {"n": len(ds[split]), "input_sha256": sha256_file(input_path),
                                     "labels_sha256": sha256_file(label_path), "fewshot_overlap_ids": sum(x["id"] in demo_ids for x in ds[split])}
    write_json(a.out / "MANIFEST.json", metadata)
    print(json.dumps(metadata["splits"], indent=2))

if __name__ == "__main__":
    main()
