"""Review generated historical CSVs against independent raw-record replay."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parent
ROOT = P.parent


def read(name):
    return json.loads((P / name).read_text(encoding="utf8"))


def rows(name):
    with (P / name).open(encoding="utf8", newline="") as f:
        return list(csv.DictReader(f))


r = read("root_independent_replay.json")
paired = read("paired_statistics.json")
for a in paired:
    b = r["transitions"]["joint" if a["family"] == "joint" else "cyclic_joint"]
    for k in ("rescue", "damage", "correct_correct", "wrong_wrong", "wrong_wrong_changed", "prediction_unchanged", "net", "exact_mcnemar_p"):
        assert a[k] == b[k], (a["family"], k)
    assert a["K1_correct"] == b["k1_correct"] and a["K4_correct"] == b["k4_correct"]
    assert np.array_equal(np.asarray(a["bootstrap_95_delta_accuracy"])*100, b["paired_bootstrap_95_delta_pp"])

raw = [json.loads(x) for x in (ROOT/"results/r11_test_selected.jsonl").read_text(encoding="utf8").splitlines() if x.strip()]
by_id = {x["id"]: x for x in raw}
tr = rows("k1_k4_paired_transitions.csv")
lp = rows("loopwise_predictions.csv")
st = rows("permutation_stability.csv")
rp = rows("cyclic_rotation_predictions.csv")
assert len(tr) == 2344 and len(lp) == 4688 and len(st) == 2344
assert len(rp) == 2*sum(len(x["cyclic_rotations"]) for x in raw) == 9374
assert len({(x["id"],x["family"]) for x in tr}) == len(tr)
assert len({(x["id"],x["family"],x["K"]) for x in lp}) == len(lp)
for x in lp:
    assert int(x["K"]) in (1,4), "Missing intermediate test scores were invented"
    k=int(x["K"]); base=by_id[x["id"]]
    name=("joint" if x["family"]=="joint" else "cyclic_joint")+str(k)+"_only"
    scores=np.asarray(json.loads(x["scores"]))
    assert np.array_equal(scores,base["joint_baseline_scores"][name])
    assert int(x["prediction"]) == base["predictions"][name]
    gold=base["gold"]
    assert float(x["gold_margin"])==float(scores[gold]-np.delete(scores,gold).max())
for k in (1,4):
    group=[x for x in st if int(x["K"])==k]
    assert len(group)==1172
    expected=r["per_depth_stability"][str(k)]
    assert sum(x["unanimous"]=="True" for x in group)==expected["unanimous_questions"]
    assert abs(np.mean([float(x["pairwise_content_agreement"]) for x in group])-expected["mean_pairwise_agreement"])<1e-15

names=["k1_k4_paired_transitions.csv","loopwise_predictions.csv","loopwise_rescue_damage.csv","permutation_stability.csv","cyclic_rotation_predictions.csv","paired_statistics.json","existing_score_parity.json","depth_readout_coverage.json"]
result=dict(status="passed_independent_CSV_and_raw_score_review", historical_test_questions=1172,
    row_counts=dict(transitions=len(tr),loopwise_predictions=len(lp),question_stability=len(st),rotation_predictions=len(rp)),
    all_historical_joint_scores_predictions_gold_margins_exact=True,
    all_transition_counts_McNemar_and_seed2021_bootstrap_exact=True,
    no_historical_K2_K3_imputation=True,
    files_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in names},
    scope="Reviews historical test outputs; fresh development/model inference has a separate admission and receipt.")
(P/"root_output_review.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
print(json.dumps(result,indent=2))
