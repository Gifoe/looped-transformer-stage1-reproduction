"""Rescore saved answers after the pre-Test content parser audit."""

import json
from pathlib import Path

import minimal
from evaluate_generation import evaluate,summarize
from task_data import load,save


def main():
    root=Path("results/development")
    out=Path("results/development_rescored")
    out.mkdir(parents=True,exist_ok=True)
    for file in sorted(root.glob("*.jsonl")):
        receipt=file.with_suffix(".json")
        if not receipt.is_file():
            continue
        old=json.loads(receipt.read_text(encoding="utf-8"))
        if old["rows_sha256"]!=minimal.sha(file):
            raise RuntimeError(f"Saved generation changed {file}")
        dataset,model,depth,mode=file.stem.split("__")
        preds=load(file)
        if mode=="content":
            info={r["sample_id"]:r for r in load(Path("data")/f"dev_{dataset}.jsonl")}
            for row in preds:
                row["metrics"]=evaluate(info[row["id"]],mode,row["prediction"],row["stop"])
        save(out/file.name,preds)
        minimal.dump(out/receipt.name,dict(plan=old["plan"],cost=old["cost"],runtime=old["runtime"],input=old["rows_sha256"],
                                           rows_sha256=minimal.sha(out/file.name),metrics=summarize(preds)))
    print("rescored",len(list(out.glob("*.json"))),flush=True)


if __name__=="__main__":
    main()
