"""Physical-layer × recurrent-depth capture, plus an optional controlled intervention.

Verified against Ouro's source: each physical layer is called once in each of
four sequential recurrent passes, and each decoder layer returns a Tensor.
K1/K2 here mean model recurrent iterations, NOT generated tokens.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class Patch:
    layer: int
    from_loop: int = 1  # 1-indexed
    to_loop: int = 2
    fraction: float = 0.0  # 0: restore K1 layer output; 1: do nothing

    def validate(self):
        if not (0 <= self.layer < 24 and 1 <= self.from_loop < self.to_loop <= 4):
            raise ValueError("Invalid physical layer/source/target recurrent step")
        if not (0 <= self.fraction <= 1):
            raise ValueError("fraction must be between 0 and 1")


class LayerRecorder:
    def __init__(self, model, positions: list[int], capture: bool, patch: Optional[Patch] = None):
        self.model = model
        self.positions = sorted(set(int(i) for i in positions))
        self.capture = bool(capture)
        self.patch = patch
        if patch is not None:
            patch.validate()
        self.calls = [0] * len(model.model.layers)
        self.snapshots = {}
        self.handles = []
        self.source_snapshot = None

    def __enter__(self):
        import torch
        self.tensor_indices = torch.as_tensor(self.positions, dtype=torch.long, device=next(self.model.parameters()).device)
        if not self.capture and self.patch is None:
            return self
        for layer, module in enumerate(self.model.model.layers):
            self.handles.append(module.register_forward_hook(self._hook(layer)))
        return self

    def _hook(self, layer):
        def hook(module, inputs, output):
            import torch
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise RuntimeError(f"Unexpected Ouro decoder-layer output type: {type(output)}")
            iteration = self.calls[layer] + 1
            self.calls[layer] = iteration
            if iteration > 4:
                raise RuntimeError("More recurrent calls than expected")
            if self.positions and self.positions[-1] >= output.shape[1]:
                raise RuntimeError("Selected position outside layer output")
            modified = output
            if self.patch is not None and layer == self.patch.layer:
                if iteration == self.patch.from_loop:
                    self.source_snapshot = output.index_select(1, self.tensor_indices).detach().clone()
                if iteration == self.patch.to_loop:
                    if self.source_snapshot is None:
                        raise RuntimeError("Missing earlier-loop donor at same physical layer")
                    if self.patch.fraction != 1.0:
                        modified = output.clone()
                        if self.patch.fraction == 0.0:
                            replacement = self.source_snapshot
                        else:
                            current = output.index_select(1, self.tensor_indices)
                            replacement = (self.source_snapshot.float() + self.patch.fraction *
                                           (current.float() - self.source_snapshot.float())).to(output.dtype)
                        modified.index_copy_(1, self.tensor_indices, replacement)
            if self.capture:
                self.snapshots[(iteration, layer)] = modified.index_select(
                    1, self.tensor_indices).detach().to("cpu", dtype=torch.bfloat16).contiguous().clone()[0]
            return modified
        return hook

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if exc_type is None and (self.capture or self.patch is not None):
            if any(count != 4 for count in self.calls):
                raise RuntimeError(f"Expected 4 calls per physical layer; got {self.calls}")
        return False

    def raw_uint16(self):
        import torch
        if not self.capture:
            return None
        tensors = [self.snapshots[(loop, layer)] for loop in range(1, 5) for layer in range(24)]
        stacked = torch.stack(tensors, dim=0).reshape(4, 24, len(self.positions), -1).contiguous()
        return stacked.view(torch.int16).numpy().view(np.uint16).copy()

    def statistics(self):
        """Descriptive all-layer recurrent deltas, no labels or assumed mechanism."""
        import torch
        if not self.capture:
            return None
        relative = np.zeros((3, 24), dtype=np.float64)
        cosine = np.zeros((3, 24), dtype=np.float64)
        for loop in range(1, 4):
            for layer in range(24):
                a = self.snapshots[(loop, layer)].float().reshape(-1)
                b = self.snapshots[(loop + 1, layer)].float().reshape(-1)
                an = float(torch.linalg.vector_norm(a))
                bn = float(torch.linalg.vector_norm(b))
                relative[loop - 1, layer] = float(torch.linalg.vector_norm(b-a)) / max(an, 1e-20)
                cosine[loop - 1, layer] = float(torch.dot(a, b)) / max(an * bn, 1e-20)
        return {"relative_l2_by_transition_and_layer": relative.tolist(),
                "cosine_by_transition_and_layer": cosine.tolist()}


def capture_positions(mode: str, input_ids: list[int], prediction_positions: list[int],
                      tokenizer, demonstration_prefix: str) -> list[int]:
    """Capture either answer, all current-task tokens, or entire context; no gold involved.

    The 'current' boundary is obtained from a common token prefix and includes
    a 16-token safety halo for possible tokenization boundary merging.
    """
    if mode == "none":
        return []
    if mode == "answer":
        return sorted(set(prediction_positions))
    if mode == "all":
        return list(range(len(input_ids)))
    if mode != "current":
        raise ValueError(f"Unknown capture mode: {mode}")
    demo_ids = tokenizer.encode(demonstration_prefix, add_special_tokens=False)
    overlap = 0
    while overlap < min(len(demo_ids), len(input_ids)) and int(demo_ids[overlap]) == int(input_ids[overlap]):
        overlap += 1
    start = max(0, overlap - 16)
    selected = list(range(start, len(input_ids)))
    return sorted(set(selected + prediction_positions))
