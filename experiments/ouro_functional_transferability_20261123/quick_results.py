"""Compact primary results for checking a running study without downloading artifacts."""

import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    root = Path("results")
    for outcome in sorted((root / "HOLDOUT").glob("**/confirm/**/SUMMARY.json")):
        print(outcome.relative_to(root))
        for ds, row in read(outcome)["evaluation"].items():
            for v, result in row["results"].items():
                y, q = (result["statistics"][name] for name in ("acc_norm", "utility"))
                print(ds, v, f'{y["native"]}->{y["controlled"]}/{row["n"]}',
                      f'{y["mean_gain"]:+.2f}pp [{y["ci95"][0]:+.2f},{y["ci95"][1]:+.2f}]',
                      f'hybrid q {q["mean_gain"]:+.4f}', f'fixed/broken {y["rescued"]}/{y["harmed"]}')
    audit = root / "PREDICTION_AUDIT/AUDIT.json"
    if audit.is_file():
        print(audit.relative_to(root))
        for family, block in read(audit).items():
            metric = block["predictors"]["utility"]
            summary = {x: dict(sign=record["exact_sign_agreement"], rho=record["spearman"],
                               rho_ci=record["spearman_question_bootstrap_ci95"])
                       for x, record in metric.items()}
            print(family, block["n_conditions"], json.dumps(summary))


if __name__ == "__main__":
    main()
