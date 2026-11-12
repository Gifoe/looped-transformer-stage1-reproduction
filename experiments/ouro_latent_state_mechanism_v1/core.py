"""Pinned Ouro runtime and label-free, native joint-option scoring primitives."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

MODEL_ID = "ByteDance/Ouro-1.4B"
REVISION = "574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
MODEL_SHA256 = "8872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"
LOOPS = 4
LAYERS = 24
HIDDEN = 2048

HERE = Path(__file__).resolve().parent
HISTORICAL = HERE.parent / "ouro_weight_space_repair_v2"
DEFAULT_MODEL_DIR = HISTORICAL / "models" / "Ouro-1.4B"
DEFAULT_DEMOS = HISTORICAL / "data" / "slcd_fixed_demonstrations.json"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def read_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def question_prompt(question: str, choices: list[str]) -> str:
    if not 2 <= len(choices) <= 8:
        raise ValueError("Expected 2–8 choices")
    return "Question: " + question + "\n" + "\n".join(
        chr(65 + i) + ". " + option for i, option in enumerate(choices)
    ) + "\nAnswer:"


def demo_prefix(demo_manifest: dict) -> str:
    documents = demo_manifest["documents"]
    if len(documents) != 25 or [x["id"] for x in documents] != demo_manifest["ids"]:
        raise ValueError("Expected historical fixed 25-shot manifest")
    rows = []
    for d in documents:
        labels = d["choices"]["label"]
        pos = labels.index(d["answerKey"])
        rows.append(question_prompt(d["question"], d["choices"]["text"]) + " " + chr(65 + pos))
    return "\n\n".join(rows) + "\n\n"


def make_requests(row: dict, demonstrations: str, order: list[int]):
    choices = [row["choices"][j] for j in order]
    prompt = demonstrations + question_prompt(row["question"], choices)
    return prompt, [(prompt, " " + chr(65 + i)) for i in range(len(order))]


def encode_pair(tokenizer, prompt: str, continuation: str):
    # Exact historical lm-evaluation-harness boundary handling, not independent encoding.
    from lm_eval.api.model import TemplateLM
    from types import SimpleNamespace
    coder = SimpleNamespace(tok_encode=lambda text, **kwargs: tokenizer.encode(text, add_special_tokens=False))
    context, target = TemplateLM._encode_pair(coder, prompt, continuation)
    if not context or not target:
        raise ValueError("Empty context or continuation")
    if len(context) + len(target) > 32768:
        raise ValueError("Sequence too long; never silently truncate")
    return [int(x) for x in context], [int(x) for x in target]


def load_ouro(model_dir: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device with BF16 support required; report hardware before changing dtype")
    model_dir = Path(model_dir).resolve()
    weight = model_dir / "model.safetensors"
    if not weight.is_file():
        raise FileNotFoundError(f"Pinned checkpoint missing: {weight}")
    digest = sha256_file(weight)
    if digest != MODEL_SHA256:
        raise RuntimeError(f"Checkpoint SHA256 mismatch: {digest}")
    torch.set_grad_enabled(False)
    torch.manual_seed(2021)
    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval().requires_grad_(False)
    model.config.use_cache = False
    model.model.config.use_cache = False
    model.config.total_ut_steps = LOOPS
    model.model.config.total_ut_steps = LOOPS
    model.model.total_ut_steps = LOOPS
    config = model.config
    if config.num_hidden_layers != LAYERS or config.hidden_size != HIDDEN:
        raise RuntimeError(f"Unexpected architecture: layers={config.num_hidden_layers}, hidden={config.hidden_size}")
    if len(model.model.layers) != LAYERS:
        raise RuntimeError("Unexpected physical layer count")
    return tok, model


def target_log_likelihoods(model, normalized_states, prediction_positions, targets):
    """Return native K1..K4 full-vocabulary FP32 log-softmax/gather/sum, label-free."""
    import torch
    target = torch.tensor(targets, dtype=torch.long, device=model.lm_head.weight.device)
    pos = torch.tensor(prediction_positions, dtype=torch.long, device=model.lm_head.weight.device)
    rows = []
    final_head = []
    with torch.inference_mode():
        for state in normalized_states:
            hidden = state.index_select(1, pos).to(model.lm_head.weight.device)
            logits = model.lm_head(hidden)[0].float()
            lp = torch.log_softmax(logits, dim=-1)
            score = float(lp.gather(-1, target[:, None]).sum().item())
            rows.append(score)
            final_head.append(logits[-1].detach().cpu().numpy().copy())
    return rows, np.stack(final_head, axis=0)
