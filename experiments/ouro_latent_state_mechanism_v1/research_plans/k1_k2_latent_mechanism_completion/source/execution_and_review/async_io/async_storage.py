"""Single CPU writer. Original frozen codec, full verification and format retained.

The GPU/model is never passed to this worker. Only already copied, detached CPU
tensors and their metadata are owned by an outstanding task. At most one task.
"""
import concurrent.futures
import hashlib
import json
import pathlib

import torch
from ab1_prefix_codec import save


def commit(directory, ident, metadata, tensors, prefix_length):
    assert all(torch.is_tensor(t) and t.device.type == "cpu" and not t.requires_grad
               for t in tensors.values())
    with torch.inference_mode():
        result = save(directory, ident, metadata, tensors, prefix_length)
    path = pathlib.Path(directory) / result["metadata_file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == result["metadata_sha256"]
    final = json.loads(path.read_text())
    return {"id": ident, "metadata_file": path.name,
            "metadata_sha256": result["metadata_sha256"],
            "tensor_raw_sha256": final["raw_tensor_sha256"]}


class SingleWriter:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1,
                                                             thread_name_prefix="evidence_writer")
        self.future = None

    def finish(self):
        if self.future is None:
            return None
        result = self.future.result()
        self.future = None
        return result

    def submit(self, directory, ident, metadata, tensors, prefix_length):
        assert self.future is None
        assert all(t.device.type == "cpu" and not t.requires_grad for t in tensors.values())
        self.future = self.executor.submit(commit, directory, ident, metadata, tensors, prefix_length)

    def close(self):
        try:
            return self.finish()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=False)
