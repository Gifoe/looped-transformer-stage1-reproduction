"""Fit on natural states, freeze on discovery, and evaluate true interventions."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from safetensors import safe_open
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ouro_latent_organization_20261118"))
from analyze_org import export, Targets
from capture_org import projection
from manifest_org import TASKS, digest, dump_line, load_lines, write_json
from models import WIDTHS, ENCODERS, METHODS, SEED, System, affine, fit_encoder, fit_transition

DOWNSTREAM = (2, 3, 4)
SIMPLE = ("identity_raw", "identity", "ridge_raw", "ridge", "rank8", "rank24", "context", "random")
NONLINEAR = ("mlp", "dictionary")
SCALES = (.3, 3., 30.)
SPLITS = ("confirm", "transfer")


class Data:
    def __init__(self, past, runs, task, width):
        self.task, self.width = task, width
        self.past = past
        legacy = load_lines(past/"natural"/task/"rows.jsonl")
        self.rows = [r for r in legacy if r["world"] in ("r", "x", "y", "d")]
        self.captured = None
        if width == 96:
            a = np.load(past/"natural"/task/"vectors.npy", mmap_mode="r")
            self.x = a[[r["i"] for r in self.rows], :2, 24].astype(np.float32)
        else:
            folder = runs/task
            if not (folder/"capture.json").is_file():
                raise RuntimeError(f"Incomplete capture {folder}")
            self.captured = load_lines(folder/"natural.jsonl")
            assert [(r["id"], r["world"], r["i"]) for r in self.rows] == [
                (r["id"], r["world"], r["legacy_i"]) for r in self.captured]
            self.x = np.load(folder/"natural_full.npy", mmap_mode="r")
        groups = {g["id"]: g for g in load_lines(past/"groups.jsonl") if g["task"] == task}
        self.groups = [groups[r["id"]] for r in self.rows]
        self.ids = {(r["id"], r["world"]): i for i, r in enumerate(self.rows)}
        self.splits = np.array([r["split"] for r in self.rows])
        self.mid = np.array([r["mid"] for r in self.rows])
        self.gold = np.array([r["gold"] for r in self.rows])
        self.context = np.zeros((len(self.rows), 384), dtype=np.float32)
        for i, r in enumerate(self.rows):
            tokens = np.array(r["ids"], dtype=np.int64)
            np.add.at(self.context[i], ((tokens*2654435761) & 0xffffffff) % 383, 1)
            self.context[i, :383] /= len(tokens)**.5
            self.context[i, -1] = len(tokens)*.01
        assert self.x.shape == (4000, 2, 5, width)

    def data_mask(self, split):
        return self.splits == split


class Scale:
    def __init__(self, fit):
        self.width = fit.shape[-1]
        self.center = fit.mean(axis=(0, 1))
        self.rms = np.sqrt(np.mean((fit-self.center)**2, axis=(0, 1, 3)))[:, None]
        assert np.min(self.rms) > 0

    def forward(self, x):
        return ((x-self.center)/self.rms).reshape(*x.shape[:-2], -1)

    def invert(self, x):
        return x.reshape(*x.shape[:-1], 5, self.width)*self.rms + self.center

    def downstream(self, x):
        return x.reshape(len(x), 5, self.width)[:, DOWNSTREAM].reshape(len(x), -1)


def key(method, encoder, dim, penalty):
    return dict(method=method, encoder=encoder, latent=dim, penalty=penalty)


def fit_system(config, train, scale, context, encoder=None):
    m, e, d, p = (config[f] for f in ("method", "encoder", "latent", "penalty"))
    if m == "identity_raw":
        return System(m)
    if m == "ridge_raw":
        return System(m, direct=affine(train[:, 0], train[:, 1]-train[:, 0], p))
    if m == "context":
        return System(m, direct=affine(context, train[:, 1], p))
    if encoder is None:
        encoder = fit_encoder(train.reshape(-1, train.shape[-1]), e, d, scale.width)
    a, b = (encoder.encode(train[:, j]) for j in (0, 1))
    return System(m, encoder, fit_transition(a, b, m, p))


def mse(x):
    return np.mean(x*x, axis=-1)


def fit_models(train, discover, scale, ctx_fit, ctx_discover, frozen=None):
    candidates = [key("identity_raw", None, None, None)]
    for m in ("ridge_raw", "context"):
        candidates.extend(key(m, None, None, p) for p in SCALES)
    if frozen is None:
        for e in ENCODERS:
            for d in WIDTHS:
                candidates.extend(key(m, e, d, p) for m in METHODS
                                  for p in (SCALES if m in ("ridge", "rank8", "rank24") else (3.,)))
    else:
        candidates.extend(frozen[m] for m in METHODS)
    winners = {}
    encoders = {}
    scores = []
    for cfg in candidates:
        m, e, d = (cfg[f] for f in ("method", "encoder", "latent"))
        if (e, d) not in encoders and e is not None:
            encoders[e, d] = fit_encoder(train.reshape(-1, train.shape[-1]), e, d, scale.width)
        model = fit_system(cfg, train, scale, ctx_fit, encoders.get((e, d)))
        pred = model(discover[:, 0], ctx_discover)
        error = float(np.mean(mse(scale.downstream(pred-discover[:, 1]))))
        if m not in winners or error < winners[m][0]:
            winners[m] = (error, cfg, model)
        scores.append({**cfg, "discover_mse": error})
    return winners, scores


def component_cases(rows, value):
    groups = collections.defaultdict(list)
    for r, v in zip(rows, value):
        groups[r["id"]].append(v)
    return {gid: np.mean(vs, axis=0) for gid, vs in groups.items()}


def grouped_rows(rows, data, metadata):
    keys = tuple(data)
    vals = component_cases(rows, np.stack([data[k] for k in keys], axis=1))
    for gid, arr in vals.items():
        yield {**metadata, "id": gid, **{k: float(a) for k, a in zip(keys, arr)}}


class Diagnostics:
    def __init__(self, data, scale, head_path):
        self.data, self.scale = data, scale
        fit = data.data_mask("fit")
        norm = scale.forward(data.x[fit]).reshape(-1, 2, 5, data.width)
        self.probes = {}
        for t in ("intermediate", "final"):
            target = Targets(data, t)
            self.probes[t] = (target, [affine(norm[:, k, r], target.y[fit], 3.)
                                      for k in (0, 1) for r in range(5)])
        self.target_ids = sorted({v for line in data.rows for v in line["target_ids"]})
        self.lookup = {t: i for i, t in enumerate(self.target_ids)}
        if data.width == 2048:
            with safe_open(head_path, framework="pt") as f:
                mat = f.get_slice("lm_head.weight")
                self.head = np.stack([mat[t:t+1].float().numpy()[0] for t in self.target_ids])
        else:
            target = np.full((fit.sum(), len(self.target_ids)), 0., dtype=np.float32)
            for i, r in enumerate(np.array(data.rows, dtype=object)[fit]):
                local = np.array(r["scores"][1], dtype=np.float32)
                target[i, [self.lookup[t] for t in r["target_ids"]]] = local-local.mean()
            self.head = affine(norm[:, 1, 4], target, 3.)

    def decoded(self, task, x, idx, loop=1, role=4, fit_loop=None):
        target, models = self.probes[task]
        role_x = x.reshape(len(x), 5, self.data.width)[:, role]
        k = loop if fit_loop is None else fit_loop
        v = models[k*5+role](role_x)
        if self.data.task == "arithmetic":
            return -np.abs(v-target.values[idx])
        return np.take_along_axis(np.pad(v, ((0, 0), (0, 1)), constant_values=-1e6),
                                  target.choice[idx], axis=1)

    def behavior(self, x, idx):
        if self.data.width == 2048:
            w = torch.from_numpy(self.head)
            native_dtype = torch.from_numpy(self.scale.invert(x)[:, 4].copy()).to(torch.bfloat16)
            # Bias-free frozen token head; BF16 logits, as in the native path.
            pred = (native_dtype @ w.T.to(torch.bfloat16)).float().numpy()
        else:
            pred = self.head(x.reshape(-1, 5, self.data.width)[:, 4])
        results = []
        for i, n in enumerate(idx):
            results.append(pred[i, [self.lookup[t] for t in self.data.rows[n]["target_ids"]]])
        return results


def margin(scores, a, b):
    return np.array([v[i]-v[j] for v, i, j in zip(scores, a, b)])


def semantic_accuracy(scores, target):
    return ((np.argmax(scores, axis=1) == target)
            & (np.sum(scores == np.max(scores, axis=1, keepdims=True), axis=1) == 1))


def natural_metrics(data, scale, systems, diag, split, output):
    test = data.data_mask(split)
    idx = np.flatnonzero(test)
    roles = scale.forward(data.x[test])
    one, two = roles[:, 0], roles[:, 1]
    ctx = data.context[idx]
    rows = [data.rows[i] for i in idx]
    result = []
    reference = mse(scale.downstream(two-one))
    for name, system in systems.items():
        pred = system(one, ctx)
        stats = {
            "state_mse": mse(scale.downstream(pred-two)),
            "delta_energy": reference,
            "k2_agreement": np.array([np.argmax(a) == np.argmax(b) for a, b in
                                     zip(diag.behavior(pred, idx),
                                         ((data.captured or data.rows)[i]["scores"][1] for i in idx))]),
        }
        for t, label in (("intermediate", data.mid), ("final", data.gold)):
            scores = diag.decoded(t, pred, idx)
            stats[t+"_k2_prediction_accuracy"] = semantic_accuracy(scores, label[idx])
        result.extend(grouped_rows(rows, stats, {"task": data.task, "split": split, "width": data.width,
                                                "method": name, "arm": "natural"}))
    role_records = []
    for name, state, k, fit_loop in (("native_K1", one, 0, 0), ("native_K2", two, 1, 1),
                                    ("K1_decoder_on_K2", two, 1, 0), ("K2_decoder_on_K1", one, 0, 1)):
        for t, label in (("intermediate", data.mid), ("final", data.gold)):
            for role in range(5):
                acc = semantic_accuracy(diag.decoded(t, state, idx, k, role, fit_loop), label[idx])
                role_records.extend(grouped_rows(rows, {"accuracy": acc},
                                    {"task": data.task, "split": split, "width": data.width,
                                     "target": t, "state": name, "role": role}))
    output.extend(role_records)
    return result


def random_response(scaled, ids, ratio, scale):
    source_energy = np.linalg.norm(scaled[:, 0], axis=1)
    vec = np.stack([np.random.default_rng(int.from_bytes(
        hashlib.sha256(("random-model/"+r["id"]+r["world"]+r["arm"]).encode()).digest()[:8], "big")
        ).standard_normal(scaled.shape[-1], dtype=np.float32) for r in ids])
    norm = np.linalg.norm(scale.downstream(vec), axis=1)
    return vec*(source_energy*ratio/np.maximum(norm, 1e-12))[:, None]


def cosine(x, y):
    norm = np.linalg.norm(x, axis=1)*np.linalg.norm(y, axis=1)
    return np.divide(np.sum(x*y, axis=1), norm, out=np.zeros(len(x)), where=norm > 1e-12)


def causal_metrics(data, scale, systems, diag, folder, case_file):
    raw = np.load(folder/"causal_full.npy", mmap_mode="r")
    rows = load_lines(folder/"causal.jsonl")
    assert len(raw) == len(rows) == 496
    frame = projection()
    if data.width == 96:
        raw = raw @ frame
    state = scale.forward(raw)
    i1, e1 = state[:, 0, 0], state[:, 1, 0]
    i2, e2 = state[:, 0, 1], state[:, 1, 1]
    real = e2-i2
    real_part = scale.downstream(real)
    idx = np.array([data.ids[r["id"], r["world"]] for r in rows])
    donor = np.array([data.ids[r["id"], r["donor"]] for r in rows])
    ctx = data.context[idx]
    natural = scale.forward(data.x[data.data_mask("fit")]).reshape(-1, 4, 2, state.shape[-1])
    pairs = natural[:, [1, 3]]-natural[:, [0, 2]]
    calibration = np.median(
        np.linalg.norm(scale.downstream(pairs[:, :, 1].reshape(-1, state.shape[-1])), axis=1)
        / np.maximum(np.linalg.norm(pairs[:, :, 0], axis=-1).reshape(-1), 1e-10))
    random = random_response(state[:, 1]-state[:, 0], rows, calibration, scale)
    true_margin = margin([r["edited_scores"][1] for r in rows], data.gold[donor], data.gold[idx]) - margin(
        [r["original_scores"][1] for r in rows], data.gold[donor], data.gold[idx])
    if data.width == 2048:
        replay = margin(diag.behavior(e2, idx), data.gold[donor], data.gold[idx]) - margin(
            diag.behavior(i2, idx), data.gold[donor], data.gold[idx])
        gap = float(np.max(np.abs(replay-true_margin)))
        assert gap < .08, gap
        print(data.task, "maximum BF16 head replay margin-delta error", gap, flush=True)
    real_sem = margin(diag.decoded("intermediate", e2, idx), data.mid[donor], data.mid[idx]) - margin(
        diag.decoded("intermediate", i2, idx), data.mid[donor], data.mid[idx])
    true_norm = np.linalg.norm(real_part, axis=1)
    result = []
    for name in (*systems, "random"):
        if name == "random":
            d = random
        else:
            estimator = systems[name]
            d = estimator(e1, ctx)-estimator(i1, ctx)
        pred = scale.downstream(d)
        norm = np.linalg.norm(pred, axis=1)
        # Anchor semantic/head responses at observed K2; the prediction is the K1-edit-induced delta only.
        actual_with_pred = i2+d
        sem = margin(diag.decoded("intermediate", actual_with_pred, idx), data.mid[donor], data.mid[idx]) - margin(
            diag.decoded("intermediate", i2, idx), data.mid[donor], data.mid[idx])
        behavioral = margin(diag.behavior(actual_with_pred, idx), data.gold[donor], data.gold[idx]) - margin(
            diag.behavior(i2, idx), data.gold[donor], data.gold[idx])
        data_fields = {
            "response_mse": mse(pred-real_part), "delta_energy": mse(real_part),
            "cosine": cosine(pred, real_part), "norm_ratio": norm/np.maximum(true_norm, 1e-12),
            "semantic_error": abs(sem-real_sem), "semantic_real_signed": real_sem,
            "semantic_pred_signed": sem, "behavior_sqerr": (behavioral-true_margin)**2,
            "behavior_real_margin": true_margin, "behavior_pred_margin": behavioral,
            "semantic_direction_agreement": (sem*real_sem > 0),
            "behavior_direction_agreement": (behavioral*true_margin > 0),
        }
        for split in SPLITS:
            for arm in ("semantic", "random") if split == "confirm" else ("semantic",):
                mask = np.array([r["split"] == split and r["arm"] == arm for r in rows])
                result.extend(grouped_rows(np.array(rows, dtype=object)[mask],
                    {k: v[mask] for k, v in data_fields.items()},
                    {"task": data.task, "split": split, "width": data.width, "method": name, "arm": arm}))
        for i, r in enumerate(rows):
            if "role_deltas_by_event" in r:
                values = {"response_cosine": cosine(pred[i:i+1], real_part[i:i+1])[0],
                          "measured_response_norm": true_norm[i], "predicted_response_norm": norm[i],
                          "measured_mid_response": real_sem[i], "predicted_mid_response": sem[i],
                          "measured_behavior_response": true_margin[i], "predicted_behavior_response": behavioral[i]}
                dump_line(case_file, {**r, "width": data.width, "method": name,
                                      **{key: float(v) for key, v in values.items()}})
    return result


def save_models(path, scale, systems):
    with path.open("wb") as f:
        pickle.dump((scale, systems), f, protocol=5)


def run(args):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    out = args.output/str(args.width)/args.task
    out.mkdir(parents=True, exist_ok=args.stage == "evaluate")
    data = Data(args.past, args.runs, args.task, args.width)
    fit, dev = map(data.data_mask, ("fit", "discover"))
    scale = Scale(data.x[fit])
    train, dev_x = map(scale.forward, (data.x[fit], data.x[dev]))
    frozen = None
    if args.width == 2048:
        first = args.output/"96"/args.task/"freeze.json"
        frozen = json.loads(first.read_text())["configurations"]
    if args.stage == "evaluate":
        assert args.width == 96
        record = json.loads((out/"freeze.json").read_text())
        for name, sha in record["implementation"].items():
            assert digest(HERE/name) == sha, name
        with (out/"models.pkl").open("rb") as f:
            scale, systems = pickle.load(f)
    else:
        winners, discovery = fit_models(train, dev_x, scale, data.context[fit], data.context[dev], frozen)
        configs = {m: a[1] for m, a in winners.items()}
        linear = min((m for m in winners if m in SIMPLE), key=lambda m: winners[m][0])
        nonlinear = min(NONLINEAR, key=lambda m: winners[m][0])
        record = {"configurations": configs, "primary_linear": linear, "primary_nonlinear": nonlinear,
                  "natural_discovery_mse": {m: a[0] for m, a in winners.items()},
                  "seed": SEED, "fit_groups": 300, "discovery_groups": 100,
                  "implementation": {name: digest(HERE/name) for name in
                                     ("fit_study.py", "capture_transition.py", "models.py", "PROTOCOL.md")}}
        write_json(out/"freeze.json", record)
        export(out/"discovery.csv", discovery)
        systems = {m: a[2] for m, a in winners.items()}
        if args.width == 96:
            save_models(out/"models.pkl", scale, systems)
        spectral = []
        for method, system in systems.items():
            enc = system.encoder
            if enc is not None:
                v = dev_x.reshape(-1, dev_x.shape[-1])
                z = enc.encode(v)
                ev = np.linalg.eigvalsh(np.cov(z.T)).clip(1e-12)
                share = ev/ev.sum()
                error = np.mean(mse(enc.decode(z)-v))
                spectral.append({"method": method, "reconstruction_mse": float(error),
                    "effective_rank": float(np.exp(-np.sum(share*np.log(share)))),
                    "latent_variance": float(ev.sum()),
                    "relative_reconstructed_variation": float(1-error/np.mean(mse(v)))})
        export(out/"encoders.csv", spectral)
    if args.stage == "freeze":
        print(args.task, args.width, record["natural_discovery_mse"], flush=True)
        return
    diag = Diagnostics(data, scale, args.model/"model.safetensors")
    readouts = []
    natural = [r for split in SPLITS for r in natural_metrics(data, scale, systems, diag, split, readouts)]
    export(out/"natural_by_group.csv", natural)
    export(out/"diagnostic_by_group.csv", readouts)
    with (out/"mechanistic_cases.jsonl").open("w", encoding="utf-8") as f:
        causal = causal_metrics(data, scale, systems, diag, args.runs/args.task, f)
    export(out/"causal_by_group.csv", causal)
    if args.width == 2048:
        save_models(out/"models.pkl", scale, {m: s for m, s in systems.items() if s.encoder is not None})
    write_json(out/"complete.json", {"seconds": time.perf_counter()-start, "width": args.width})
    print(args.task, args.width, record["natural_discovery_mse"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("past", type=Path)
    p.add_argument("runs", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("task", choices=TASKS)
    p.add_argument("width", type=int, choices=(96,2048))
    p.add_argument("--stage", choices=("freeze", "evaluate", "report"), default="report")
    run(p.parse_args())
