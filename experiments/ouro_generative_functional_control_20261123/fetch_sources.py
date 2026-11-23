"""Stage the pinned public sources for reconstruction of the experiment data."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import shutil
from urllib.request import urlopen


def file_url(dataset, revision, path):
    return f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{path}"


def sha(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def fetch(path, url, expected):
    if path.exists():
        if sha(path) != expected:
            raise ValueError(f"Existing source fingerprint differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".part")
    with urlopen(url, timeout=60) as response, scratch.open("wb") as sink:
        shutil.copyfileobj(response, sink, length=1024 * 1024)
    if sha(scratch) != expected:
        raise ValueError(f"Downloaded source fingerprint differs: {path}")
    scratch.replace(path)
    print(path, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--inputs", type=Path, default=Path("inputs"))
    p.add_argument("--published-data", type=Path, required=True)
    p.add_argument("--extras", action="store_true", help="Fetch the two expansion shards after initial pool preparation")
    a = p.parse_args()
    lock = json.loads((a.published_data / "DATA_LOCK.json").read_text(encoding="utf-8"))
    expected = lock["input_build_hashes"]
    smol, srev = lock["revisions"]["smoltalk"].split("@")
    magpie = "smol-magpie-ultra"
    if a.extras:
        scale = json.loads((a.published_data / "SCALE_LOCK.json").read_text(encoding="utf-8"))
        for name, value in scale["sources"].items():
            index = int(Path(name).stem[-2:])
            fetch(a.inputs / name, file_url(smol, srev, f"data/{magpie}/train-{index:05d}-of-00006.parquet"), value)
        return
    for name, folder, splits in (("squad", "plain_text", ("train", "validation")),
                                 ("gsm8k", "main", ("train", "test")),
                                 ("no_robots", "data", ("train", "test"))):
        dataset, rev = lock["revisions"][name].split("@")
        for split in splits:
            leaf = f"{name}_{split}.parquet"
            fetch(a.inputs / leaf, file_url(dataset, rev, f"{folder}/{split}-00000-of-00001.parquet"), expected[leaf])
    for subset in (magpie, "smol-constraints", "smol-rewrite", "smol-summarize"):
        for split in ("train", "test"):
            shards = 6 if subset == magpie and split == "train" else 1
            leaf = f"{subset}_{split}.parquet"
            fetch(a.inputs / leaf, file_url(smol, srev, f"data/{subset}/{split}-00000-of-{shards:05d}.parquet"), expected[leaf])
    for prefix, dataset, rev, folder in (
        ("arc_easy", "allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Easy"),
        ("upstream_arc_challenge", "allenai/ai2_arc", "210d026faf9955653af8916fad021475a3f00453", "ARC-Challenge"),
        ("obqa", "allenai/openbookqa", "8c4888010de9b23a2d136ebd31d5f5ea9f3d90fa", "main"),
        ("qasc", "allenai/qasc", "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070", "data"),
    ):
        for split in ("train", "test", "validation"):
            leaf = f"{prefix}_{split}.parquet"
            fetch(a.prior / "inputs" / leaf, file_url(dataset, rev, f"{folder}/{split}-00000-of-00001.parquet"), expected[leaf])
    fetch(a.prior / "inputs" / "csqa_train.jsonl", "https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl",
          "58ffa3c8472410e24b8c43f423d89c8a003d8284698a6ed7874355dedd09a2fb")


if __name__ == "__main__":
    main()
