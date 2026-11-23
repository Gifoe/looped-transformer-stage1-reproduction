"""Allow official eight-choice QASC in the unchanged depth/source trainer."""

from pathlib import Path

import depth_holdout as shared


original_validate = shared.canonical_data


def validate(path):
    if path.name != "qasc_train.jsonl":
        return original_validate(path)
    rows = shared.read_rows(path)
    seen = set()
    for row in rows:
        labels, content = (row["choices"][key] for key in ("label", "text"))
        if (not isinstance(row["id"], str) or row["id"] in seen
                or not isinstance(row["question"], str) or not row["question"].strip()
                or len(labels) != 8 or len(set(labels)) != 8 or len(content) != 8
                or row["answerKey"] not in labels or any(not isinstance(t, str) or not t.strip() for t in content)):
            raise ValueError(f"Invalid official QASC row {row['id']}")
        seen.add(row["id"])
    if not rows:
        raise ValueError("Empty QASC training set")
    return rows


class QascDepthTransfer(shared.DepthTransfer):
    def __init__(self, args):
        super().__init__(args)
        if args.source != "qasc":
            raise ValueError("Use the regular shared trainer for non-QASC sources")
        self.source_hashes[Path(__file__).name] = shared.sha(Path(__file__))


if __name__ == "__main__":
    shared.canonical_data = validate
    shared.DepthTransfer = QascDepthTransfer
    shared.main()
