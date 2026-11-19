"""Check the frozen item IDs and exclusion lists without reading any labels."""
import argparse
from itertools import combinations
import json
from pathlib import Path


def ids(path):
    return set(json.loads(path.read_text(encoding="utf-8"))["exclude_ids"])


def verify(manifest_path, targeted_path, old_openbook_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = {"fit": 512, "dev": 128, "diagnostic": 256, "transfer": 384}
    assert manifest["label_data_read"] is False
    assert set(manifest["cohorts"]) == set(counts)
    seen = {key: {row["id"] for row in rows} for key, rows in manifest["cohorts"].items()}
    clusters = {key: {row["cluster"] for row in rows} for key, rows in manifest["cohorts"].items()}
    for key, expected in counts.items():
        assert len(manifest["cohorts"][key]) == len(seen[key]) == len(clusters[key]) == expected, key
    for a, b in combinations(counts, 2):
        assert clusters[a].isdisjoint(clusters[b]), (a, b)
    for a, b in combinations(("fit", "dev", "diagnostic"), 2):
        assert seen[a].isdisjoint(seen[b]), (a, b)
    arc_exposure = ids(targeted_path)
    old_openbook = ids(old_openbook_path)
    assert len(arc_exposure) == 351 and len(old_openbook) == 35
    assert not ((seen["dev"] | seen["diagnostic"]) & arc_exposure)
    assert seen["transfer"].isdisjoint(old_openbook)
    return {
        "counts": counts,
        "raw_label_files_read": 0,
        "arc_dev_or_diagnostic_intersection_with_targeted_exposure": 0,
        "openbook_transfer_intersection_with_historical_stem_duplicate_ids": 0,
        "arc_fit_historical_targeted_exposure_rows": len(seen["fit"] & arc_exposure),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("arc_targeted_exposure", type=Path)
    parser.add_argument("openbook_exposure", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest, args.arc_targeted_exposure, args.openbook_exposure), indent=2))
