"""One CPU writer per isolated worker; private CAS, stage-relative manifests.

Reuses the original byte-verifying codec. Per-worker CAS prevents concurrent
exclusive-create collisions. Final case metadata is committed only after full
reconstruction from the rebased stage-relative chunk paths succeeds.
"""
import concurrent.futures
import hashlib
import json
import os
import pathlib

import torch
from ab1_prefix_codec import save, load


def commit(stage, namespace, ident, metadata, tensors, prefix):
    assert all(t.device.type == "cpu" and not t.requires_grad for t in tensors.values())
    stage = pathlib.Path(stage)
    private = stage / namespace
    assert private.resolve().is_relative_to(stage.resolve())
    with torch.inference_mode():
        result = save(private, ident, metadata, tensors, prefix)
        internal = private / result["metadata_file"]
        rebased = json.loads(internal.read_text())
        for value in rebased["tensor_manifest"].values():
            for chunk in value["chunks"]:
                chunk["file"] = (pathlib.Path(namespace) / chunk["file"]).as_posix()
        rebased["storage_namespace"] = namespace
        reconstructed = load(rebased, stage)
        assert set(reconstructed) == set(tensors)
        for name, expected in tensors.items():
            assert torch.equal(reconstructed[name].reshape(-1).view(torch.uint8),
                               expected.contiguous().reshape(-1).view(torch.uint8)), name
    destination = stage / result["metadata_file"]
    payload = (json.dumps(rebased, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with destination.open("xb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    assert destination.read_bytes() == payload
    return {"id": ident, "metadata_file": destination.name,
            "metadata_sha256": hashlib.sha256(payload).hexdigest(),
            "tensor_raw_sha256": rebased["raw_tensor_sha256"]}


class Writer:
    def __init__(self):
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.pending = None

    def finish(self):
        if self.pending is None:
            return None
        value = self.pending.result()
        self.pending = None
        return value

    def submit(self, *args):
        assert self.pending is None
        self.pending = self.pool.submit(commit, *args)

    def close(self):
        try:
            return self.finish()
        finally:
            self.pool.shutdown(wait=True, cancel_futures=False)
