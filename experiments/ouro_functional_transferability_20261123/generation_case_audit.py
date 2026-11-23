"""Post-hoc case-only diagnostic using frozen greedy output strings."""

from hashlib import file_digest
import json
from pathlib import Path
import re

import numpy as np


def case_recover(row):
    if row["valid"]:
        return row["option"]
    if row["invalid_category"] == "lowercase_option":
        match = re.match(r"\s*([a-z])\b", row["text"])
        if match is not None and match[1].upper() in row["options"]:
            return match[1].upper()
    return None


def main():
    root = Path("results/TRANSFER_GENERATION")
    reports = {}
    for ds in ("arc_round2_confirm", "csqa_round2_confirm"):
        parsed = {}
        for match in sorted(root.glob(f"{ds}__*__k4.jsonl")):
            name = match.name.split("__")[1]
            with match.open("rb") as stream:
                digest = file_digest(stream, "sha256").hexdigest()
            rows = [json.loads(line) for line in match.read_text(encoding="utf-8").splitlines()]
            parsed[name] = (rows, [case_recover(row) for row in rows], digest)
        reference, reference_answers, _ = parsed["native"]
        native = np.array([guess == row["gold"] for guess,row in zip(reference_answers,reference,strict=True)])
        rng = np.random.default_rng(261123173)
        draws = rng.integers(len(native), size=(6000,len(native)))
        reports[ds] = {}
        for model,(rows,answer,digest) in parsed.items():
            assert [x["id"] for x in rows] == [x["id"] for x in reference]
            correct = np.array([guess == row["gold"] for guess,row in zip(answer,rows,strict=True)])
            delta = correct.astype(float) - native.astype(float)
            reports[ds][model] = dict(n=len(rows), sha256=digest, original_valid=sum(r["valid"] for r in rows),
                original_correct=sum(r["correct"] for r in rows), case_relaxed_valid=sum(x is not None for x in answer),
                case_relaxed_correct=int(correct.sum()), rescue=int((delta>0).sum()), damage=int((delta<0).sum()),
                paired_delta_pp=float(100*delta.mean()), paired_ci95_pp=(100*np.quantile(delta[draws].mean(1),[.025,.975])).tolist())
    artifact = dict(status="post_hoc_format_only_diagnostic", original_scoring_unchanged=True, datasets=reports)
    (root/"CASE_AUDIT.json").write_text(json.dumps(artifact,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(reports,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
