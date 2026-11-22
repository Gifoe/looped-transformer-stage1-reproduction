"""Synthetic semantic specificity and frozen zero-shot CommonsenseQA transfer."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

import e_design as design


DEPTHS = (1, 2, 3, 4)
SEMANTIC_DEPTH = 4
FIXED_DOSE = 0.45
ROTATION_SEED = 2026112327
CSQA_REVISION = "author random-split dev export"
CSQA_SOURCE = "https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl"
CSQA_SHA256 = "3210497fdaae614ac085d9eb873dd7f4d49b6f965a93adadc803e1229fd8a02a"
CSQA_COUNT = 1221
LAYERS = {"early": range(0, 8), "middle": range(8, 16), "late": range(16, 24)}


def digest(path):
    fingerprint = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024**2), b""):
            fingerprint.update(chunk)
    return fingerprint.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def lock_manifest(out, value):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "LOCK.json"
    if path.exists():
        if read_json(path) != json.loads(json.dumps(value)):
            raise RuntimeError(f"Resume manifest differs from {path}")
    else:
        design.save(path, value)


def artifact(path, sha):
    if digest(path) != sha:
        raise RuntimeError(f"Artifact digest mismatch in {path}")
    return path


def baseline(args):
    receipt = read_json(args.rank8_freeze)
    entry = receipt["arms"]["linear8_t1"]
    path = artifact(args.rank8_freeze.parent / entry["file"], entry["sha256"])
    return path, float(entry["dose"])


def frozen(args, transfer=False):
    receipt = read_json(args.freeze)
    requested = receipt["primary"] if transfer else receipt["class_representatives"]["adaptive"]
    entry = receipt["weights"][requested]
    path = artifact(args.freeze.parent / "weights" / entry["file"], entry["sha256"])
    return dict(name=requested, path=path, form=entry["form"],
                primary_selection_eligible=receipt["selection_eligible"])


def manifest(args, rank8, dose, chosen):
    return dict(
        code={name: digest(Path(__file__).parent / name)
              for name in ("experiment_e.py", "e_design.py", "e_stats.py", "controller.py", "c_common.py")},
        runtime={"archive": str(args.archive.resolve()), "model": str(args.model.resolve()),
                 "previous": str(args.previous.resolve())},
        rank8=dict(file=str(rank8), sha256=digest(rank8), dose=dose,
                   receipt_sha256=digest(args.rank8_freeze)),
        current_freeze_sha256=digest(args.freeze),
        controller=dict(name=chosen["name"], form=chosen["form"], sha256=digest(chosen["path"]),
                        primary_selection_eligible=chosen["primary_selection_eligible"]),
        batch_size=args.batch_size,
    )


def synthetic_encode(tokenizer, rows):
    items = []
    repeated = defaultdict(set)
    candidate_tokens = defaultdict(set)
    prefix_lengths = []
    for i, row in enumerate(rows):
        text = row["prompt"]
        prefix = tokenizer.encode(text, add_special_tokens=False)
        alternatives = []
        for word in row["choices"]["text"]:
            full = tokenizer.encode(text + " " + word, add_special_tokens=False)
            if full[:-1] != prefix:
                raise RuntimeError(f"Expected a single answer token with an unchanged prefix in {row['id']}")
            candidate_tokens[word].add(full[-1])
            alternatives.append(full)
        positions = design.evidence_positions(tokenizer, row, prefix)
        repeated[row["group"], row["variant"]].add((len(prefix), *positions))
        prefix_lengths.append(len(prefix))
        items.append((alternatives[0], 1, i, 0, alternatives[1][-1]))
    if any(len(x) != 1 for x in repeated.values()) or any(len(x) != 1 for x in candidate_tokens.values()):
        raise RuntimeError("Expected identical evidence positions and candidate tokenization in matched cells")
    audit = dict(candidate_token_id={k: next(iter(v)) for k, v in sorted(candidate_tokens.items())},
                 matched_evidence_positions_verified=len(repeated),
                 matched_position_index={f"{group}:{variant}": next(iter(values))
                                         for (group, variant), values in repeated.items()},
                 prompt_token_length_counts=dict(sorted(Counter(prefix_lengths).items())))
    return sorted(items, key=lambda x: len(x[0])), audit


def batches(out, tag, items, rows, limit, score):
    chunks = out / "chunks" / tag
    chunks.mkdir(parents=True, exist_ok=True)
    values = {}
    seconds = 0.0
    forwards = 0
    for start in range(0, len(items), limit):
        subset = items[start:start + limit]
        identity = [[rows[item[2]]["id"], item[3]] for item in subset]
        path = chunks / f"{start // limit:05d}.json"
        if path.exists():
            record = read_json(path)
            if record["identity"] != identity:
                raise RuntimeError(f"Resume chunk identity differs in {path}")
        else:
            started = time.perf_counter()
            output, calls = score(subset)
            record = dict(identity=identity, scores=output, calls=calls, seconds=time.perf_counter() - started)
            design.save(path, record)
        for item, value in zip(subset, record["scores"], strict=True):
            values[item[2], item[3]] = value
        seconds += record["seconds"]
        forwards += record["calls"]
    if len(values) != len(items):
        raise RuntimeError(f"Duplicate candidate identities in {tag}")
    timing = dict(batch_size=limit, rows=len(rows), items=len(items), forward_calls=forwards, wall_s=seconds)
    print(tag, round(seconds, 2), forwards, flush=True)
    return values, timing


def make_orientation(orientation, torch):
    from minimal import rms

    class FixedEdit:
        kind = "linear"

        def delta(self, hidden, previous=None):
            return FIXED_DOSE * rms(hidden) * orientation

    return FixedEdit()


def clamp_locations(path):
    if path is None:
        return set()
    frozen = read_json(path)
    return {(4, layer, "attention") for layer in LAYERS[frozen["locations"]["attention"]]}


def score_synthetic(old, model, subset, net, dose, clamp, torch):
    from minimal import forward, rms

    packed = old.pack([item[:4] for item in subset])
    if clamp:
        from c_common import forward as clamped_forward

        _, reference = clamped_forward(model, packed, branch_layers=clamp)
        update = net.delta(reference["raw"][3]) * dose
        hidden, trace = clamped_forward(model, packed, anchor=3, update=update,
                                       branch_layers=clamp, clamp_layers=clamp, reference=reference)
        # Small differences in realized RMS are expected after the model dtype cast.
        dose_realized = rms(trace["edits"]) / rms(reference["raw"][3])
        if not bool((dose_realized - FIXED_DOSE).abs().max() < .01):
            raise RuntimeError("Clamped intervention does not match the prescribed injection RMS")
        calls = 2
    else:
        hidden, _ = forward(model, packed, SEMANTIC_DEPTH, net, dose)
        calls = 1
    alternate = torch.tensor([item[4] for item in subset], device=packed[-1].device)
    targets = torch.stack((packed[-1], alternate), dim=-1)
    logits = model.lm_head(hidden).float()
    prob = logits.log_softmax(-1).gather(1, targets)
    return prob.tolist(), calls


def semantic_generation(args, rows, arms, model, tokenizer):
    import torch
    from generation import Arm, classify, generate_batch, model_next

    output = []
    summary = {}
    costs = {}
    selected = design.generation_subset(rows)
    for mode in ("content", "label"):
        for name in ("native", "rank8", "shared_fixed", "shared_learned"):
            path = args.out / f"GENERATION_{mode}_{name}.jsonl"
            receipt_path = path.with_suffix(".cost.json")
            if path.exists() and receipt_path.exists():
                records = read_rows(path)
                receipt = read_json(receipt_path)
                if ([row["id"] for row in records] != [row["id"] for row in selected]
                        or digest(path) != receipt["sha256"]):
                    raise RuntimeError(f"Generation records differ from the frozen subset in {path}")
            else:
                started = time.perf_counter()
                forward_calls, padded_tokens = 0, 0
                net, dose = arms[name]
                arm = Arm("native") if net is None else Arm("minimal", net, dose)
                records = []
                for start in range(0, len(selected), args.batch_size):
                    subset = selected[start:start + args.batch_size]
                    prompts = [row["prompt"].replace("option text", "option letter") if mode == "label"
                               else row["prompt"] for row in subset]
                    encoded = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]
                    tokens, stops, calls, padded = generate_batch(
                        tokenizer, encoded,
                        lambda packed, last: model_next(model, None, arm, SEMANTIC_DEPTH, packed, last),
                        5, "cuda",
                    )
                    forward_calls += calls
                    padded_tokens += padded
                    for row, prompt, token, stop in zip(subset, prompts, tokens, stops, strict=True):
                        text = tokenizer.decode(token, skip_special_tokens=True)
                        if mode == "label":
                            label, invalid, strict = classify(text, row["options"])
                            option = row["options"].get(label)
                        else:
                            words = "|".join(re.escape(word) for word in row["options"].values())
                            match = re.match(rf"^\s*({words})\b", text, re.IGNORECASE)
                            option = match[1].casefold() if match else None
                            strict = text.strip().rstrip(".").casefold() == option
                            invalid = None if option else ("unparsed_content" if text.strip() else "empty")
                        records.append(dict(id=row["id"], group=row["group"], family=row["family"], prompt=prompt,
                                            method=name, mode=mode, stop=stop, tokens=token, text=text,
                                            gold_content=row["gold_content"], gold_label=row["answerKey"],
                                            parsed_content=option, valid=option is not None, strict=strict,
                                            invalid_category=invalid, correct=option == row["gold_content"]))
                torch.cuda.synchronize()
                design.save(path, records, jsonl=True)
                receipt = dict(wall_s=time.perf_counter() - started, sha256=digest(path),
                               forwards=forward_calls, padded_input_tokens=padded_tokens, depth=SEMANTIC_DEPTH)
                design.save(receipt_path, receipt)
            summary[f"{mode}_{name}"] = dict(
                n=len(records), valid=sum(row["valid"] for row in records),
                strict=sum(row["strict"] for row in records), correct=sum(row["correct"] for row in records),
                complete_groups=sum(all(r["correct"] for r in group) for group in design.grouped(records).values()),
                groups=len(design.grouped(records)), total_generated_tokens=sum(len(row["tokens"]) for row in records),
                invalid_categories=dict(Counter(r["invalid_category"] for r in records if not r["valid"])),
            )
            costs[f"{mode}_{name}"] = receipt
            output.extend(records)
    design.save(args.out / "GENERATION_PREDICTIONS.jsonl", output, jsonl=True)
    design.save(args.out / "GENERATION_SUMMARY.json", summary)
    design.save(args.out / "GENERATION_COST.json", costs)


def semantic(args, old, model, tokenizer, runtime_info, rank8, dose, chosen):
    import numpy as np
    import torch
    from controller import load_controller, prior_axis
    from e_stats import fit_lexical_bias, semantic_summary
    from minimal import load_net, rms

    train, heldout = design.population("train"), design.population("heldout")
    populations = design.audit(train, heldout)
    train_items, train_tokens = synthetic_encode(tokenizer, train)
    heldout_items, heldout_tokens = synthetic_encode(tokenizer, heldout)
    clamp = clamp_locations(args.c_lock)
    direction = prior_axis(args.axis, "cuda")
    generator = torch.Generator(device="cpu").manual_seed(ROTATION_SEED)
    orthogonal = torch.randn(direction.shape, generator=generator).to(direction.device)
    orthogonal -= direction * (direction * orthogonal).mean()
    orthogonal /= rms(orthogonal)
    lock_manifest(args.out, dict(
        **manifest(args, rank8, dose, chosen), experiment="semantic", depth=SEMANTIC_DEPTH,
        fixed_dose=FIXED_DOSE, axis_sha256=digest(args.axis), seed_rotation=ROTATION_SEED,
        axis_orientation_cosine=float((direction * orthogonal).mean()),
        c_train_lock_sha256=digest(args.c_lock) if args.c_lock else None,
        clamped_locations=sorted(clamp), synthetic_generation=args.generate,
        primary="Canonical same-candidate fact reversal x independent option remap. K4.",
        helper_runtime=runtime_info,
    ))
    design.write_population(args.out / "population", train, heldout)
    design.save(args.out / "TOKEN_AUDIT.json", dict(train=train_tokens, heldout=heldout_tokens))
    arms = dict(
        native=(None, 1.0), rank8=(load_net(rank8), dose),
        shared_fixed=(make_orientation(direction, torch), 1.0),
        shared_learned=(load_controller(chosen["path"]), 1.0),
        reversed=(make_orientation(-direction, torch), 1.0),
        random=(make_orientation(orthogonal, torch), 1.0),
    )
    if clamp:
        arms.update(fixed_clamped=arms["shared_fixed"], random_clamped=arms["random"])
    training, training_cost = batches(
        args.out, "train_native", train_items, train, args.batch_size,
        lambda subset: score_synthetic(old, model, subset, None, 1.0, set(), torch),
    )
    lexical = fit_lexical_bias(train, [training[i, 0] for i in range(len(train))])
    design.save(args.out / "LEXICAL_TRAIN_PREDICTIONS.jsonl",
                [dict(id=row["id"], scores=training[i, 0]) for i, row in enumerate(train)], jsonl=True)
    design.save(args.out / "LEXICAL_FIT.json", lexical)
    costs = dict(train_native=training_cost)
    scores = {}
    for name, (net, injection_dose) in arms.items():
        selected_clamp = clamp if name.endswith("_clamped") else set()
        values, costs[name] = batches(
            args.out, name, heldout_items, heldout, args.batch_size,
            lambda subset: score_synthetic(old, model, subset, net, injection_dose, selected_clamp, torch),
        )
        scores[name] = [values[i, 0] for i in range(len(heldout))]
    for i, row in enumerate(heldout):
        row["scores"] = {arm: score[i] for arm, score in scores.items()}
        row["scores"]["lexical"] = [x + lexical["weights"][w] for x, w in
                                    zip(row["scores"]["native"], row["choices"]["text"], strict=True)]
        if not all(np.isfinite(score).all() for score in row["scores"].values()):
            raise RuntimeError(f"Non-finite candidate score for {row['id']}")
    summary = semantic_summary(heldout)
    cancellation = summary["populations"]["pooled"]["results"]["canonical"]["margin"]
    if abs(cancellation["lexical"]["mean"] - cancellation["native"]["mean"]) > 1e-10:
        raise RuntimeError("Fixed candidate biases must cancel at the matched continuous reversal endpoint")
    design.save(args.out / "PREDICTIONS.jsonl", heldout, jsonl=True)
    design.save(args.out / "SUMMARY.json", dict(
        **summary, populations_audit=populations,
        branch_test="frozen Experiment C Train attention band, K4" if clamp else "No frozen branch tested in E",
        scoring="Full one-token option text, two equally long candidate strings. Raw and character-normalized rankings coincide.",
    ))
    design.save(args.out / "COST.json", costs)
    if args.generate:
        semantic_generation(args, heldout, arms, model, tokenizer)


def fetch_csqa(args):
    cache = args.csqa_file or args.out / "commonsenseqa_validation.jsonl"
    if not cache.exists():
        if args.csqa_file:
            raise RuntimeError(f"Requested local CommonsenseQA file is missing: {cache}")
        temporary = cache.with_suffix(".download")
        request = Request(CSQA_SOURCE, headers={"User-Agent": "ouro-functional-controllability/1.0"})
        with urlopen(request, timeout=45) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 64):
                output.write(chunk)
        artifact(temporary, CSQA_SHA256)
        temporary.replace(cache)
    return artifact(cache, CSQA_SHA256)


def transfer(args, old, model, tokenizer, runtime_info, rank8, dose, chosen, source):
    import numpy as np
    from controller import load_controller
    from e_stats import transfer_summary
    from minimal import forward, load_net

    source_receipt = dict(source=CSQA_SOURCE, revision=CSQA_REVISION, sha256=CSQA_SHA256,
                          split="Validation", n=CSQA_COUNT)
    lock_manifest(args.out, dict(**manifest(args, rank8, dose, chosen), experiment="transfer",
                                 source=source_receipt, evaluated_depths=DEPTHS, helper_runtime=runtime_info,
                                 scoring="Archived candidate answer text zero-shot, no readout edits or task fit"))
    raw = read_rows(source)
    if len(raw) != CSQA_COUNT or len({row["id"] for row in raw}) != CSQA_COUNT:
        raise RuntimeError("CommonsenseQA Validation must contain all 1221 unique examples")
    questions = []
    for row in raw:
        choices = row["question"]["choices"]
        labels = [option["label"] for option in choices]
        if labels != list("ABCDE") or row["answerKey"] not in labels:
            raise RuntimeError(f"Unexpected author data schema in {row['id']}")
        questions.append(dict(id=row["id"], question=row["question"]["stem"], answerKey=row["answerKey"],
                              choices=dict(label=labels, text=[option["text"] for option in choices])))
    items, metadata = old.encode(tokenizer, questions)
    items.sort(key=lambda x: len(x[0]))
    arms = {"native": (None, 1.0), "rank8": (load_net(rank8), dose),
            "frozen": (load_controller(chosen["path"]), 1.0)}
    timing = {}
    for name, (net, scale) in arms.items():
        for depth in DEPTHS:
            tag = f"{name}_k{depth}"

            def score(subset):
                packed = old.pack(subset)
                hidden, _ = forward(model, packed, depth, net, scale)
                logits = model.lm_head(hidden).float()
                return old.scores(logits, packed[1].cpu().numpy(), packed[-1], len(subset)).tolist(), 1

            values, timing[tag] = batches(args.out, tag, items, metadata, args.batch_size, score)
            old.record(metadata, items, {tag: [values[item[2], item[3]] for item in items]})
    for row in metadata:
        chars = np.array(row["characters"])
        row["predictions"] = {}
        for tag, values in row["scores"].items():
            if len(values) != 5 or not np.isfinite(values).all():
                raise RuntimeError(f"Incomplete transfer scores for {row['id']}")
            row["predictions"][tag] = dict(acc=int(np.argmax(values)), acc_norm=int(np.argmax(values / chars)))
    design.save(args.out / "PREDICTIONS.jsonl", metadata, jsonl=True)
    design.save(args.out / "SUMMARY.json", dict(dataset=source_receipt, controller=chosen["name"],
                                              primary_selection_eligible=chosen["primary_selection_eligible"],
                                              **transfer_summary(metadata)))
    design.save(args.out / "COST.json", timing)
    design.save(args.out / "TRANSFER_STATUS.json", dict(status="complete", controller=chosen["name"],
                                                        primary_selection_eligible=chosen["primary_selection_eligible"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("design", "semantic", "transfer"))
    parser.add_argument("--out", type=Path, required=True)
    for name in ("archive", "model", "previous", "freeze", "rank8-freeze", "axis", "c-lock", "csqa-file"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.phase == "design":
        design.write_population(args.out, design.population("train"), design.population("heldout"))
        return
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    required = ["model", "archive", "previous", "freeze", "rank8_freeze"]
    if args.phase == "semantic":
        required.append("axis")
    for key in required:
        if getattr(args, key) is None:
            parser.error("--" + key.replace("_", "-") + " is required")
    chosen = frozen(args, transfer=args.phase == "transfer")
    args.out.mkdir(parents=True, exist_ok=True)
    rank8, dose = baseline(args)
    source = fetch_csqa(args) if args.phase == "transfer" else None
    import torch
    from minimal import runtime

    old, model, tokenizer, info = runtime(args.archive, args.previous, args.model)
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        if args.phase == "semantic":
            semantic(args, old, model, tokenizer, info, rank8, dose, chosen)
        else:
            transfer(args, old, model, tokenizer, info, rank8, dose, chosen, source)


if __name__ == "__main__":
    main()
