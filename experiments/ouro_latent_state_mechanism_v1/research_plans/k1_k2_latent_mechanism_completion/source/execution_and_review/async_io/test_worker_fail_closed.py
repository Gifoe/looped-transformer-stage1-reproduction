"""Synthetic error/inference-tensor QA, distinct from real scientific evidence."""
import hashlib
import json
import pathlib
import sys
import tempfile

root = pathlib.Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / "code"), str(root / "deps"), str(root / "async_io")]
import torch
import async_storage

torch.set_num_threads(1)
results = {}
writer = async_storage.SingleWriter()
try:
    writer.submit(pathlib.Path("unused"), "synthetic_meta", {},
                  {"x": torch.empty(1, device="meta")}, 0)
except AssertionError:
    results["non_cpu_tensor"] = "REJECTED_BEFORE_SUBMIT"
else:
    raise AssertionError("Non-CPU tensor accepted")
finally:
    writer.close()
original = async_storage.save


def fail(*args, **kwargs):
    raise OSError("SYNTHETIC_STORAGE_FAILURE")


async_storage.save = fail
writer = async_storage.SingleWriter()
writer.submit(pathlib.Path("unused"), "synthetic_error", {}, {"x": torch.zeros(1)}, 0)
try:
    writer.finish()
except OSError as error:
    assert "SYNTHETIC_STORAGE_FAILURE" in str(error)
    results["storage_failure"] = "PROPAGATED_TO_MAIN_THREAD"
else:
    raise AssertionError("Storage error swallowed")
try:
    writer.close()
except OSError:
    pass
async_storage.save = original
with tempfile.TemporaryDirectory(prefix="async_cpu_inference_fixture_") as temporary:
    destination = pathlib.Path(temporary)
    with torch.inference_mode():
        tensors = {"x": torch.arange(32, dtype=torch.bfloat16).reshape(1, 32).clone()}
    assert tensors["x"].is_inference()
    writer = async_storage.SingleWriter()
    writer.submit(destination, "synthetic_inference_tensor", {"scope": "SYNTHETIC_QA_ONLY"}, tensors, 0)
    item = writer.close()
    assert item["id"] == "synthetic_inference_tensor"
    results["inference_tensor_cross_thread"] = "FULL_ORIGINAL_CODEC_WRITE_AND_RELOAD_PASS"
receipt = {"status": "SYNTHETIC_QA_PASS", "checks": results,
           "script_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
           "scope": "Not real GPU model replay or a scientific result"}
(root / "async_io/SYNTHETIC_FAIL_CLOSED_QA.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt))
