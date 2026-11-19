"""Prepare verified benchmark inputs for a fresh reproduction."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from urllib.error import URLError
from urllib.request import urlopen


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def obtain(dest, relative, expected, source):
    if dest.exists():
        if sha256(dest) != expected:
            raise ValueError(f"Existing file has an unexpected SHA-256: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    causes = []
    try:
        for host in source:
            try:
                with urlopen(f"{host}/{relative}", timeout=20) as response, tmp.open("wb") as out:
                    shutil.copyfileobj(response, out)
                if sha256(tmp) == expected:
                    tmp.replace(dest)
                    return
                causes.append(f"{host}: unexpected SHA-256")
            except (URLError, TimeoutError) as exc:
                causes.append(f"{host}: {exc}")
        raise RuntimeError(f"Unable to verify {dest}. " + " | ".join(causes))
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--historical-test", action="store_true")
    parser.add_argument("--mirror", help="Optional mirror of Hugging Face model and dataset URLs")
    args = parser.parse_args()
    source = ["https://huggingface.co"]
    if args.mirror:
        source.append(args.mirror.rstrip("/"))
    manifest = json.loads(Path(__file__).with_name("input_manifest.json").read_text(encoding="utf-8"))
    arc = manifest["arc_challenge"]
    arc_base = f'datasets/{arc["repository"]}/resolve/{arc["revision"]}/{arc["subset"]}'
    splits = ("train", "validation", "test") if args.historical_test else ("train", "validation")
    for split in splits:
        target = args.destination / "arc_challenge" / f"{split}.parquet"
        obtain(target, f"{arc_base}/ai2_arc-{split}.parquet", arc["splits"][split]["sha256"], source)
        print(target)
    hotpot = manifest["hotpotqa"]
    target = args.destination / "hotpot_distractor_validation.parquet"
    relative = f'datasets/{hotpot["repository"]}/resolve/{hotpot["revision"]}/{hotpot["file"]}'
    obtain(target, relative, hotpot["sha256"], source)
    print(target)


if __name__ == "__main__":
    main()
