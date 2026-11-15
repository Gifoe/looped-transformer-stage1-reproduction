import collections
import json
import pathlib

root = pathlib.Path(__file__).resolve().parents[1]
path = root / "parallel/benchmark_v2/serial/worker0_METADATA_DIFFERENCES.json"
data = json.loads(path.read_text())
differences = []


def walk(a, b, key):
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys(), key
        for name in a:
            walk(a[name], b[name], key + "/" + name)
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), key
        for index, (left, right) in enumerate(zip(a, b)):
            walk(left, right, key + "/" + str(index))
    elif a != b:
        differences.append({"path": key, "actual": a, "expected": b})


for name, item in data.items():
    walk(item["actual"], item["expected"], name)
address_keys = {"postadd_argument_object_id", "postadd_argument_data_ptr"}
other = [item for item in differences if item["path"].split("/")[-1] not in address_keys]
receipt = {"n_total_differences": len(differences),
           "leaf_key_counts": dict(collections.Counter(item["path"].split("/")[-1] for item in differences)),
           "n_nonaddress_differences": len(other), "nonaddress_examples": other[:5],
           "scope": "Cross-process object IDs and allocation addresses are runtime provenance, not numerical tensors. They remain retained in actual records."}
(root / "parallel/V2_METADATA_DIFFERENCE_DIAGNOSIS.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
