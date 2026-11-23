"""Scratch Rank-1 depth, position, and task transfer on a frozen Ouro."""

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import torch

from forward_control import forward_control


@dataclass(frozen=True)
class Group:
    name: str
    depths: tuple[int, ...]
    anchor: int | None = None


DEPTH_GROUPS = tuple(Group(name, depths) for name, depths in (
    ("k2", (2,)), ("k3", (3,)), ("k4", (4,)), ("k23", (2, 3)), ("k24", (2, 4)), ("k234", (2, 3, 4))))
POSITIONS = tuple(Group(f"p{j}", (4,), j) for j in (1, 2, 3))
SUITES = {"depth": DEPTH_GROUPS, "position": POSITIONS, "source": (DEPTH_GROUPS[-1],)}
DEFAULT_SEED = 261123137
BOOT_SEED = 261123324
CAP = .45


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    temp = path.with_suffix(".temporary")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def save_rows(path, rows):
    temp = path.with_suffix(".temporary")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
    temp.replace(path)


def require_lock(path, value):
    normalized = json.loads(json.dumps(value))
    if path.exists():
        if read(path) != normalized:
            raise RuntimeError(f"Existing study lock differs: {path}")
    else:
        save(path, normalized)


def read_rows(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def named(values):
    result = {}
    for value in values:
        key, separator, path = value.partition("=")
        if (separator != "=" or re.fullmatch(r"[a-z][a-z_0-9]*", key) is None or not path or key in result):
            raise ValueError(f"Use unique NAME=PATH input names: {value!r}")
        result[key] = Path(path)
    return result


def canonical_data(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
    else:
        rows = read_rows(path)
    ids = set()
    for row in rows:
        if not isinstance(row["id"], str) or row["id"] in ids:
            raise ValueError(f"Duplicate or invalid question ID: {row['id']!r}")
        ids.add(row["id"])
        labels, texts = (row["choices"][key] for key in ("label", "text"))
        if (not isinstance(row["question"], str) or not row["question"].strip()
                or len(labels) != len(set(labels)) or len(labels) != len(texts) or not 2 <= len(labels) <= 5
                or row["answerKey"] not in labels or any(not isinstance(t, str) or not t.strip() for t in texts)):
            raise ValueError(f"Noncanonical question data for {row['id']}")
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def select_train(rows, source, n):
    if len(rows) < n:
        raise ValueError(f"Requested {n} training questions from {source}, available {len(rows)}")
    if any(row.get("source_split", "train") not in ("train", "official/train") for row in rows):
        raise ValueError("A training input contains non-Train examples")
    def order(group):
        return sorted(group, key=lambda r: hashlib.sha256(f"scratch-rank1-v1/{source}/{r['id']}".encode()).digest())
    if source == "mixed":
        arc, csqa = ([r for r in rows if r.get("dataset")==name] for name in ("arc", "csqa"))
        n_arc, n_csqa = (n+1)//2, n//2
        if len(arc)+len(csqa) != len(rows) or len(arc)<n_arc or len(csqa)<n_csqa or n<2:
            raise ValueError("Mixed source requires balanced annotated ARC and CSQA Train examples")
        return order(order(arc)[:n_arc] + order(csqa)[:n_csqa])
    if any(row.get("dataset", source) != source for row in rows):
        raise ValueError("Training rows contain a different dataset than --source")
    return order(rows)[:n]


def question_updates(n, batch, steps, seed):
    """Ensure identical full mini-batches across groups, with seeded epoch wrap."""
    rng = np.random.default_rng(seed)
    order = []
    while len(order) < batch * steps:
        order.extend(rng.permutation(n).tolist())
    return [order[i:i+batch] for i in range(0, batch*steps, batch)]


def tensor_digest(net):
    out = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        x = value.detach().cpu().contiguous().numpy()
        out.update(name.encode() + str(x.shape).encode() + x.dtype.str.encode() + x.tobytes())
    return out.hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def profile(scores, meta):
    raw = np.asarray(scores, dtype=np.float64)
    norm = raw / np.asarray(meta["characters"], dtype=np.float64)
    gold = meta["gold"]
    raw_margin = raw[gold] - np.delete(raw, gold).max()
    norm_margin = norm[gold] - np.delete(norm, gold).max()
    l1, l2 = raw / 3, norm * 16
    utility = .5 * (l1[gold] - np.logaddexp.reduce(l1)) + .5 * (l2[gold] - np.logaddexp.reduce(l2))
    return dict(acc=int(raw.argmax() == gold), acc_norm=int(norm.argmax() == gold),
                margin=float(raw_margin), margin_norm=float(norm_margin), utility=float(utility))


def paired(rows, reference, method, draws):
    before, after = [[profile(row["scores"][name], row) for row in rows] for name in (reference, method)]
    result = {}
    for metric in before[0]:
        x, y = [np.asarray([r[metric] for r in table]) for table in (before, after)]
        is_accuracy = metric.startswith("acc")
        delta = y - x
        scale = 100 if is_accuracy else 1
        result[metric] = dict(
            native=int(x.sum()) if is_accuracy else float(x.mean()),
            controlled=int(y.sum()) if is_accuracy else float(y.mean()),
            mean_gain=float(scale*delta.mean()), ci95=(scale*np.quantile(delta[draws].mean(1), [.025,.975])).tolist(),
            rescued=int((delta>0).sum()), harmed=int((delta<0).sum()), tied=int((delta==0).sum()))
    return result


class DepthTransfer:
    def __init__(self, a):
        sys.path.insert(0, str(a.minimal.resolve()))
        import minimal
        import train

        self.a, self.minimal, self.old_train = a, minimal, train
        self.old, self.model, self.tokenizer, self.runtime = minimal.runtime(a.archive, a.previous, a.model)
        self.model.eval().requires_grad_(False)
        self.device = next(self.model.parameters()).device
        if self.model.config.hidden_size != 2048 or self.model.model.config.num_hidden_layers != 24:
            raise RuntimeError("Expected the frozen 24-layer 2,048-channel Ouro")
        self.root = a.out / a.source / a.suite / f"s{a.seed}"
        self.root.mkdir(parents=True, exist_ok=True)
        requested = a.groups.split(",") if a.groups else [g.name for g in SUITES[a.suite]]
        self.groups = [g for g in SUITES[a.suite] if g.name in requested]
        if len(requested) != len(self.groups):
            raise ValueError("Invalid or repeated group names for this suite")
        paths = (Path(__file__), Path(__file__).with_name("forward_control.py"), Path(minimal.__file__), Path(train.__file__))
        self.source_hashes = {p.name: sha(p) for p in paths}

    @torch.enable_grad()
    def fit_one(self, group, items, metadata, plan, lock_sha):
        a = self.a
        folder = self.root / group.name
        folder.mkdir(exist_ok=True)
        ckpt, receipt = folder / "controller.pt", folder / "TRAIN.json"
        if receipt.exists():
            outcome = read(receipt)
            if (outcome["checkpoint_sha256"], outcome["lock_sha256"]) != (sha(ckpt), lock_sha):
                raise RuntimeError(f"Broken training receipt: {receipt}")
            return outcome
        torch.manual_seed(a.seed)
        net = self.minimal.Minimal("linear", 1, CAP).to(self.device)
        initialization = tensor_digest(net)
        if torch.count_nonzero(net.out.weight) != 0:
            raise RuntimeError("Effective initial rank-one operator must be zero")
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=.03)
        calls = token_loops = candidate_tokens = targets = 0
        records = []
        start = time.perf_counter()
        for step, qids in enumerate(plan):
            begin = time.perf_counter()
            depth = group.depths[step % len(group.depths)]
            # Holdout is enforced by the executed per-step schedule and logged separately.
            if depth not in group.depths:
                raise RuntimeError(f"Configured holdout support was violated by K{depth}")
            entries = [item for q in qids for item in items[q]]
            packed = self.old.pack(entries)
            opt.zero_grad(set_to_none=True)
            hidden, _ = forward_control(self.model, packed, depth, anchor_loop=group.anchor, controller=net)
            lp, logits = self.old_train.logprobs(self.model, hidden, packed[-1])
            del logits
            loss, _ = self.old_train.ranking(lp, packed[1], metadata, qids)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            if not bool(torch.isfinite(grad)) or (step == 0 and float(grad) <= 0):
                raise RuntimeError(f"Invalid training gradient in {group.name} step {step}")
            opt.step()
            inputs = packed[0].numel()
            supervised = packed[-1].numel()
            calls += 1
            candidate_tokens += inputs
            token_loops += inputs * depth
            targets += supervised
            records.append(dict(step=step, depth=depth, anchor=group.anchor or depth-1,
                question_ids=[metadata[q]["id"] for q in qids], candidates=len(entries), candidate_targets=supervised,
                padded_input_tokens=inputs, loss=float(loss.detach()), gradient_l2=float(grad),
                wall_ms=1000*(time.perf_counter()-begin)))
            if (step+1) % 40 == 0 or step == 0:
                print(self.a.source,self.a.suite,group.name,step+1,depth,records[-1]["loss"],flush=True)
        torch.cuda.synchronize()
        if set(r["depth"] for r in records) != set(group.depths):
            raise RuntimeError(f"Depth coverage was incomplete in {group.name}")
        net.eval()
        portable = dict(initial_state_sha256=initialization, source=self.a.source, suite=self.a.suite, group=group.name,
                        seed=a.seed, trained_depths=list(group.depths), anchor_loop=group.anchor, steps=len(records),
                        rank_one_parameters=sum(x.numel() for x in net.parameters()), lock_sha256=lock_sha)
        self.minimal.save_net(ckpt, net, portable)
        outcome = dict(**portable, checkpoint_sha256=sha(ckpt), actual_train_depth_counts=dict(Counter(
            f"k{r['depth']}" for r in records)), fit_finished_utc=timestamp(), wall_seconds=time.perf_counter()-start,
            backbone_calls=calls, padded_input_tokens=candidate_tokens, padded_input_token_loops=token_loops,
            scored_target_tokens=targets, actual_unique_questions=len({i for r in records for i in r["question_ids"]}),
            step_records=records)
        save(receipt, outcome)
        return outcome

    def fit(self):
        a = self.a
        all_train = canonical_data(a.train_data)
        raw = select_train(all_train, a.source, a.sample_n)
        items, metadata = self.old_train.partitions(self.old, self.tokenizer, raw)
        plan = question_updates(len(raw), a.batch, a.steps, a.seed)
        plan_fingerprint = hashlib.sha256(np.asarray(plan, dtype="<i8").tobytes()).hexdigest()
        manifest = dict(data_sha256=sha(a.train_data), runtime=self.runtime, source_hashes=self.source_hashes,
            source=a.source, suite=a.suite, seed=a.seed, training_candidates=len(raw), training_ids=[x["id"] for x in raw],
            training_source_counts=dict(Counter(x.get("dataset",a.source) for x in raw)),
            plan_sha256=plan_fingerprint, optimizer=dict(type="AdamW", lr=a.lr, weight_decay=.03, clip_grad_norm=1.,
                updates=a.steps, questions_per_update=a.batch), init="Pytorch seeded input factor, zero write factor",
            loss="Prior full-answer ranking loss, 0.5 CE(raw logp/3) + 0.5 CE(16 logp/characters)",
            cap=CAP, stats_per_optimizer_round="wall time, loss, gradient, input/target tokens, question IDs",
            groups=[vars(c) for c in SUITES[a.suite]])
        path = self.root / "TRAIN_LOCK.json"
        require_lock(path, manifest)
        weight_map_path = self.root / "TRAIN_PACKET.json"
        packet = read(weight_map_path) if weight_map_path.exists() else dict(
            train_lock_sha256=sha(path), source=a.source, suite=a.suite, seed=a.seed, checkpoints={})
        if packet["train_lock_sha256"] != sha(path):
            raise RuntimeError("Training packet does not refer to the current manifest")
        for group in self.groups:
            r = self.fit_one(group, items, metadata, plan, sha(path))
            packet["checkpoints"][group.name] = {key:r[key] for key in (
                "checkpoint_sha256","initial_state_sha256","trained_depths","anchor_loop","steps","fit_finished_utc",
                "wall_seconds","actual_train_depth_counts","actual_unique_questions")}
            save(weight_map_path, packet)
        if len({m["initial_state_sha256"] for m in packet["checkpoints"].values()}) != 1:
            raise RuntimeError("The matched groups do not share initial factors")

    def evaluation_inputs(self):
        """Verify every checkpoint and the precommitted evaluation and predictions hashes before parsing labels."""
        a = self.a
        train_path = self.root / "TRAIN_LOCK.json"
        manifest = read(train_path)
        packet_path = self.root / "TRAIN_PACKET.json"
        packet = read(packet_path)
        if (packet["train_lock_sha256"] != sha(train_path) or manifest["source_hashes"] != self.source_hashes
                or manifest["runtime"]["model_sha256"] != self.runtime["model_sha256"]):
            raise RuntimeError("Training/evaluation model or code provenance differs")
        checkpoints, digests = {}, {}
        for group in self.groups:
            path = self.root / group.name / "controller.pt"
            digest = sha(path)
            if digest != packet["checkpoints"][group.name]["checkpoint_sha256"]:
                raise RuntimeError(f"Checkpoint hash changed for {group.name}")
            checkpoints[group.name], digests[group.name] = path, digest
        eval_files = named(a.eval)
        data_hashes = {name: sha(path) for name, path in eval_files.items()}
        forecast = None
        if a.forecast_lock:
            forecast_data = read(a.forecast_lock)
            if not set(digests.values()) <= set(forecast_data["controller_sha256s"]):
                raise RuntimeError("A controller was not previously committed in the forecast lock")
            if any(forecast_data["evaluation_sha256s"].get(k) != h for k,h in data_hashes.items()):
                raise RuntimeError("A held-out input file does not match the forecast lock")
            if sha(a.predictions) != forecast_data["predictions_sha256"]:
                raise RuntimeError("Predictive artifact is not the precommitted file")
            forecast = dict(lock_sha256=sha(a.forecast_lock), predictions_sha256=sha(a.predictions))
        return manifest, checkpoints, digests, eval_files, data_hashes, forecast

    def archive_baselines(self, metadata, native_file, depths):
        result = [dict(row,scores={}) for row in metadata]
        if native_file is None:
            return result, set(depths)
        raw = read_rows(native_file)
        historical = {r["id"]:r for r in raw}
        if len(raw) != len(historical) or not {q["id"] for q in metadata} <= historical.keys():
            raise RuntimeError(f"Archived native scores cannot cover these question IDs: {native_file}")
        for row in result:
            old = historical[row["id"]]
            if any(row[k] != old[k] for k in ("labels","gold","answer_tokens","characters","scored_input_tokens")):
                raise RuntimeError(f"Archived native score protocol differs on {row['id']}")
        missing = set()
        for depth in depths:
            key = f"native_k{depth}"
            if not all(key in historical[r["id"]]["scores"] for r in result):
                missing.add(depth)
                continue
            for row in result:
                values = historical[row["id"]]["scores"][key]
                if len(values) != len(row["labels"]):
                    raise RuntimeError(f"Native archived candidate count changed for {row['id']}")
                row["scores"][key] = values
        return result, missing

    def eval_native(self, items, metadata, depths, native_file):
        records, missing = self.archive_baselines(metadata, native_file, depths)
        work = calls = 0
        t0 = time.perf_counter()
        if missing:
            max_depth = max(missing)
            for offset in range(0, len(items), self.a.eval_batch):
                batch = items[offset:offset+self.a.eval_batch]
                packed = self.old.pack(batch)
                _, trace = forward_control(self.model, packed, max_depth, capture=missing)
                for k in sorted(missing):
                    score = self.score(trace["post"][k][packed[1],packed[2]], packed, len(batch))
                    self.old.record(records, batch, {f"native_k{k}":score})
                calls += 1
                work += packed[0].numel() * max_depth
        return records, dict(calls=calls, padded_input_token_loops=work, seconds=time.perf_counter()-t0,
            recomputed_depths=sorted(missing), archived_depths=sorted(set(depths)-missing))

    def score(self, terminal, packed, count):
        return self.old.scores(self.model.lm_head(terminal).float(), packed[1].cpu().numpy(), packed[-1], count)

    def eval_controller(self, items, metadata, net, depth, anchor, tag):
        records = [dict(row,scores={}) for row in metadata]
        per_write, calls, work = [], 0, 0
        start = time.perf_counter()
        for i in range(0, len(items), self.a.eval_batch):
            subset = items[i:i+self.a.eval_batch]
            packed = self.old.pack(subset)
            h, trace = forward_control(self.model, packed, depth, anchor_loop=anchor, controller=net)
            score = self.score(h, packed, len(subset))
            self.old.record(records, subset, {tag:score})
            scale = self.minimal.rms(trace["anchor"])
            pre = self.minimal.rms(net.out(net.encode(trace["anchor"]))) / scale
            actual = self.minimal.rms(trace["delta"]) / scale
            per_write.append(torch.cat((pre,actual,(pre>CAP).float()),dim=1).cpu().numpy())
            calls += 1
            work += packed[0].numel()*depth
        dose = np.concatenate(per_write)
        return records, dict(calls=calls, padded_input_token_loops=work, seconds=time.perf_counter()-start,
            writes=len(dose), rms_ratio_before_clip_mean=float(dose[:,0].mean()),
            rms_ratio_actual_mean=float(dose[:,1].mean()), rms_ratio_actual_quantiles=np.quantile(dose[:,1],[.05,.5,.95]).tolist(),
            clipping_fraction=float(dose[:,2].mean()), anchor_loop=anchor if anchor is not None else (depth-1 if depth>1 else None),
            anchor_layer=21 if depth==1 and anchor is None else None)

    @torch.inference_mode()
    def evaluate(self):
        a = self.a
        manifest, ckpts, digests, inputs, data_hashes, forecast = self.evaluation_inputs()
        native_inputs = named(a.native)
        if not native_inputs.keys() <= inputs.keys():
            raise ValueError("Archived native scoring must refer to a declared evaluation dataset")
        native_hashes = {name:sha(p) for name,p in native_inputs.items()}
        depths = tuple(a.eval_depths or ((4,) if a.suite=="position" else (2,3,4,5,6)))
        folder = self.root / a.phase
        if a.eval_tag:
            folder = folder / a.eval_tag
        folder.mkdir(parents=True,exist_ok=True)
        if a.phase == "discovery" and any((self.root/"confirm").glob("**/EVAL_LOCK.json")):
            raise RuntimeError("Discovery is sealed after confirmation has started")
        own_ids = set(manifest["training_ids"])
        rows = {name:canonical_data(path) for name,path in inputs.items()}
        for name, dataset in rows.items():
            if set(q["id"] for q in dataset) & own_ids:
                raise RuntimeError(f"Evaluation overlaps this checkpoint's Train sample: {name}")
        require_lock(folder/"EVAL_LOCK.json",dict(role=a.phase, forecast=forecast, dataset_sha256=data_hashes,
            native_archive_sha256=native_hashes, checkpoint_sha256=digests, batch=a.eval_batch, depths=depths,
            questionnaire_ids={name:[q["id"] for q in rr] for name,rr in rows.items()}))
        nets = {name:self.minimal.load_net(path).requires_grad_(False) for name,path in ckpts.items()}
        report = {}
        cells = [(k, p) for k in depths for p in ((1,2,3) if a.suite=="position" else (None,))]
        for name, dataset in rows.items():
            items, metadata = self.old.encode(self.tokenizer, dataset)
            items.sort(key=lambda record:len(record[0]))
            dataset_tags = sorted({q["dataset"] for q in dataset if "dataset" in q})
            native_scores = folder / f"{name}_native.jsonl"
            native_receipt = native_scores.with_suffix(".json")
            if native_receipt.exists():
                native_cost = read(native_receipt)
                if native_cost["sha256"] != sha(native_scores):
                    raise RuntimeError(f"Native score fragments changed: {native_scores}")
                result = read_rows(native_scores)
            else:
                result, native_cost = self.eval_native(items,metadata,depths,native_inputs.get(name))
                save_rows(native_scores,result)
                native_cost["sha256"] = sha(native_scores)
                save(native_receipt,native_cost)
            boot = np.random.default_rng(BOOT_SEED).integers(0,len(result),(10_000,len(result)),dtype=np.int32)
            costs, points = {}, {}
            for group in self.groups:
                for depth, anchor in cells:
                    tag = f"{group.name}_k{depth}" + (f"_p{anchor}" if anchor is not None else "")
                    path = folder / f"{name}_{tag}.jsonl"
                    receipt = path.with_suffix(".json")
                    if receipt.exists():
                        costs[tag] = read(receipt)
                        if costs[tag]["scores_sha256"] != sha(path):
                            raise RuntimeError(f"Partial controller prediction changed: {path}")
                        pred = read_rows(path)
                    else:
                        pred, costs[tag] = self.eval_controller(items,metadata,nets[group.name],depth,anchor,tag)
                        save_rows(path,pred)
                        costs[tag]["scores_sha256"] = sha(path)
                        save(receipt,costs[tag])
                    for old,new in zip(result,pred,strict=True):
                        if any(old[k]!=new[k] for k in ("id","labels","gold","characters","answer_tokens","scored_input_tokens")):
                            raise RuntimeError(f"Unpaired score row: {name}, {old['id']}")
                        old["scores"][tag] = new["scores"][tag]
                    if group.anchor is not None:
                        membership = "seen_position" if group.anchor==anchor else "unseen_position"
                    elif depth in group.depths:
                        membership = "seen_depth"
                    elif min(group.depths) < depth < max(group.depths):
                        membership = "unseen_within_range"
                    else:
                        membership = "unseen_above_range" if depth>max(group.depths) else "unseen_below_range"
                    points[tag] = dict(depth_membership=membership, outside_historical_k2_k4=depth>=5 or depth==1,
                        evaluation_depth=depth, controller_source=a.source, checkpoint_sha256=digests[group.name],
                        statistics=paired(result,f"native_k{depth}",tag,boot))
                    print(a.phase,name,tag,points[tag]["statistics"]["acc_norm"]["controlled"],flush=True)
            file = folder / f"{name}_PREDICTIONS.jsonl"
            save_rows(file,result)
            report[name] = dict(n=len(result), dataset_tags=dataset_tags, results=points, costs=costs, native_cost=native_cost,
                                complete_candidate_scores_sha256=sha(file))
        save(folder/"SUMMARY.json",dict(study_role=a.phase, checkpoints=digests, forecast=forecast, evaluation=report))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("train","discovery","confirm"),required=True)
    p.add_argument("--source",choices=("arc","csqa","obqa","qasc","mixed"),required=True)
    p.add_argument("--suite",choices=SUITES,default="depth")
    for flag in ("archive","previous","minimal","model","out"):
        p.add_argument(f"--{flag}",type=Path,required=True)
    p.add_argument("--train-data", type=Path)
    p.add_argument("--expected-train-sha256")
    p.add_argument("--sample-n",type=int,default=1119)
    p.add_argument("--steps",type=int,default=282)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--eval-batch",type=int,default=16)
    p.add_argument("--lr",type=float,default=.001)
    p.add_argument("--seed",type=int,default=DEFAULT_SEED)
    p.add_argument("--groups")
    p.add_argument("--eval",action="append",default=[])
    p.add_argument("--eval-tag",help="Optional distinct result namespace under the evaluation phase")
    p.add_argument("--native",action="append",default=[])
    p.add_argument("--eval-depths",nargs="+",type=int,choices=range(1,7))
    p.add_argument("--forecast-lock",type=Path)
    p.add_argument("--predictions",type=Path)
    a=p.parse_args()
    if a.phase=="train" and (not a.train_data or a.eval or a.native or a.forecast_lock or a.predictions or a.eval_tag):
        p.error("Training requires --train-data and never accepts evaluation inputs")
    if a.phase!="train" and (not a.eval or a.train_data or a.expected_train_sha256):
        p.error("Evaluation requires one or more --eval NAME=PATH and never accepts Train input")
    if a.phase=="confirm" and not a.forecast_lock:
        p.error("Confirmation requires precommitted --forecast-lock and --predictions")
    if bool(a.forecast_lock)!=bool(a.predictions):
        p.error("Forecast lock and predictions file are required together")
    if min(a.sample_n,a.batch,a.steps,a.eval_batch)<1 or a.steps<a.batch or not 0<a.lr<1:
        p.error("Use a positive feasible training and evaluation budget")
    if a.steps % 6 and a.suite in ("depth","source"):
        p.error("Use a multiple of six steps for exact balance over two or three training depths")
    if a.eval_depths and (len(set(a.eval_depths))!=len(a.eval_depths) or (a.suite=="position" and a.eval_depths!=[4])):
        p.error("The position suite is fixed at K4; other suites need unique depths")
    if a.eval_tag and re.fullmatch(r"[a-z][a-z_0-9]*",a.eval_tag) is None:
        p.error("Use a lowercase alphanumeric evaluation tag")
    if a.expected_train_sha256 and sha(a.train_data)!=a.expected_train_sha256:
        p.error("Training data hash differs from the preregistration")
    study=DepthTransfer(a)
    if a.phase=="train":
        study.fit()
    else:
        study.evaluate()


if __name__=="__main__":
    main()
