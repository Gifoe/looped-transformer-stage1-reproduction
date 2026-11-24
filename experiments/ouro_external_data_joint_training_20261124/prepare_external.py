"""Build source-locked question-level pools before creating response views."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest
import json
from pathlib import Path
import re

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from task_data import encode_target, hash_order, identity, load, multiple_choice, save


QUOTAS = {"csqa": 1000, "obqa": 1000, "gsm8k": 2000, "scienceqa": 1000}
REVISIONS = {"csqa": "tau/commonsense_qa@d90214517943631510b39b7d2503b7ea93ec09c2",
             "obqa": "allenai/openbookqa@8c4888010de9b23a2d136ebd31d5f5ea9f3d90fa",
             "gsm8k": "openai/gsm8k@740312add88f781978c0658806c59bc2815b9866"}
VISUAL = re.compile(r"\b(?:picture|image|photograph|photo|figure|graph|chart|diagram|map|illustration|drawing|shown|pictured|depicted|shaded)\b|\b(?:above|below|left|right)\s+(?:figure|image|diagram|panel|hand side)\b", re.I)
NUMERIC = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?\Z")
CALCULATOR = re.compile(r"<<[^<>]*>>")
MAX_TOTAL = 640
MAX_RESPONSE = 512
DEV_PER_SOURCE = 360


def sha(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parquet(path):
    return pq.read_table(path).to_pylist()


def normalize_question(row):
    question = (row.get("source_original_question") or row.get("question") or row.get("question_stem")
                or row.get("candidate", {}).get("question"))
    if isinstance(question, dict):
        question = question.get("stem")
    if question:
        return identity(question)
    if row.get("requests"):
        context = row["requests"][0]["args"][0]
        questions = re.findall(r"(?:^|\n)Question:\s*([^\n]+)", context)
        body = re.sub(r"^The following are multiple choice questions[^\n]*\n+", "", context)
        return identity(questions[-1] if questions else re.split(r"\n\s*(?:A[.)]|Answer:)", body)[0])
    return identity(row.get("instruction") or row.get("context") or "")


def read_rows(path):
    return parquet(path) if path.suffix == ".parquet" else load(path)


def shingles(normalized):
    words = normalized.split()
    return frozenset(zip(words, words[1:], words[2:])) if len(words) >= 6 else frozenset()


class Corpus:
    def __init__(self):
        self.samples = []
        self.exact = {}
        self.index = defaultdict(list)

    def add(self, text, identifier):
        if not text or text in self.exact:
            return
        i = len(self.samples)
        terms = shingles(text)
        self.samples.append((identifier, terms))
        self.exact[text] = identifier
        for word in terms:
            self.index[word].append(i)

    def match(self, text):
        if text in self.exact:
            return self.exact[text], 1.0
        terms = shingles(text)
        counts = Counter(j for s in terms for j in self.index.get(s, ()))
        best = (None, 0.)
        for j, shared in counts.items():
            identifier, other = self.samples[j]
            sim = shared / (len(terms) + len(other) - shared)
            if sim >= .8 and sim > best[1]:
                best = identifier, sim
        return best


def choice_record(row, dataset):
    source_id = str(row["id"])
    prefix = f"{dataset}/{source_id}"
    question = row.get("question_stem", row.get("question"))
    opts = row.get("choices")
    if isinstance(question, dict):
        opts, question = question["choices"], question["stem"]
    if isinstance(opts, list):
        opts = {key: [choice[key] for choice in opts] for key in ("label", "text")}
    gold = str(row["answerKey"])
    normalized = [identity(t) for t in opts["text"]]
    if not question or len(set(normalized)) != len(normalized) or not all(normalized) or gold not in opts["label"]:
        return None
    candidate = dict(id=prefix, question=question, choices=opts, answerKey=gold)
    return dict(dataset=dataset, sample_id=prefix, source_id=source_id, question=question, candidate=candidate)


def science_record(pid, row):
    question, hint, solution = row["question"], row["hint"], row["solution"]
    opts = row["choices"]
    labels = [chr(65 + i) for i in range(len(opts))]
    i = row["answer"]
    if row["image"] or VISUAL.search(" ".join([question, hint, solution, *opts])):
        return None, "visual"
    if not solution.strip() or not 0 <= i < len(opts):
        return None, "missing_explanation_or_answer"
    options = [identity(text) for text in opts]
    if len(set(options)) != len(opts) or not all(options) or any(x in {"all of the above", "both a and b"} for x in options):
        return None, "ambiguous_options"
    text = f" {identity(solution)} "
    answer = options[i]
    stated = re.findall(r"\b(?:the\s+)?(?:correct\s+)?(?:answer|option)\s+is\s+\(?([A-Ea-e])(?:\s*[.)]|\s*$)", solution)
    if any(letter.upper() != labels[i] for letter in stated):
        return None, "contradictory_explicit_answer"
    if f" {answer} " not in text and labels[i] not in [letter.upper() for letter in stated]:
        return None, "answer_not_supported_by_surface_check"
    stem = question if not hint.strip() else f"{question}\nHint: {hint}"
    built = choice_record(dict(id=pid, question=stem, answerKey=labels[i], choices=dict(text=opts, label=labels)), "scienceqa")
    return dict(built, question=question, explanation=solution.strip()), None


def views(record):
    dataset = record["dataset"]
    if dataset == "gsm8k":
        prefix = "Solve the problem step by step. End with the final answer after ####."
        final = "Solve the problem and give only the final number after ####."
        return {name: dict(context=record["question"], dataset=dataset, sample_id=record["sample_id"], task_type=f"gsm_{name}",
                           prompt=f"Question: {record['question']}\n{prompt}\nAnswer:", response=answer,
                           references=[record["answer"]])
                for name, (prompt, answer) in {"process": (prefix, record["solution"]), "final": (final, "#### " + record["answer"])}.items()}
    candidate = record["candidate"]
    short = multiple_choice(candidate, dataset, shuffle=dataset != "scienceqa")
    if dataset != "scienceqa":
        return {"short": short}
    instruction = "Explain briefly, then give the full text of the correct option after Final answer:."
    prompt = short["prompt"].replace(short["instruction"], instruction)
    answer = f"{record['explanation']}\nFinal answer: {short['answer_content']}"
    return {"short": short, "explanation": dict(short, prompt=prompt, response=answer, task_type="mc_explanation")}


def length_check(record, tok, stats):
    formed = views(record)
    for row in formed.values():
        try:
            prefix, full, n = encode_target(tok, row)
        except (AssertionError, ValueError):
            stats["token_boundary"] += 1
            return None
        if len(full) > MAX_TOTAL or n > MAX_RESPONSE:
            stats["over_length"] += 1
            return None
        row.update(prompt_tokens=len(prefix), response_tokens=n, input_tokens=len(full)-1)
    return dict(record, views=formed)


def disjoint_components(records, reservations, audit):
    index = Corpus()
    parents = list(range(len(records)))
    reserved = {}

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, record in enumerate(records):
        text = identity(record["question"])
        match, sim = reservations.match(text)
        if match is not None:
            reserved[i] = (match, sim)
        if text in index.exact:
            parents[i] = root(index.exact[text])
            continue
        terms = shingles(text)
        possible = Counter(j for term in terms for j in index.index.get(term, ()))
        for j, count in possible.items():
            linked, other = index.samples[j]
            if count / (len(terms) + len(other) - count) >= .8:
                a, b = root(i), root(linked)
                parents[max(a, b)] = min(a, b)
        index.add(text, i)
    rejected = {root(i): hit for i, hit in reserved.items()}
    components = defaultdict(list)
    for i, record in enumerate(records):
        c = root(i)
        if c in rejected:
            match, sim = rejected[c]
            audit.append(dict(id=record["sample_id"], reason="reserved_component", reference=match, similarity=sim))
        else:
            components[c].append(record)
    picked = []
    for same in components.values():
        kept = min(same, key=lambda x: hash_order(x["sample_id"], "external/component"))
        picked.append(kept)
        audit.extend(dict(id=row["sample_id"], reason="training_component", reference=kept["sample_id"]) for row in same if row is not kept)
    return picked


def select(pool, quota, dataset):
    salt = "external/selection/" + dataset
    if dataset != "gsm8k":
        return sorted(pool, key=lambda x: hash_order(x["sample_id"], salt))[:quota]
    strata = sorted(pool, key=lambda x: x["views"]["process"]["response_tokens"])
    cuts = [strata[i*len(strata)//4:(i+1)*len(strata)//4] for i in range(4)]
    return [x for cohort in cuts for x in sorted(cohort, key=lambda x: hash_order(x["sample_id"], salt))[:quota//4]]


def build(args):
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    audit, stats = [], {name: Counter() for name in QUOTAS}
    refs = Corpus()
    provenance = {}
    for path in sorted(set(args.reserve)):
        rows = read_rows(path)
        for i, row in enumerate(rows):
            refs.add(normalize_question(row), f"{path.name}:{row.get('sample_id', row.get('id', i))}")
        provenance[str(path)] = dict(rows=len(rows), sha256=sha(path))
    records = []
    revisions = dict(REVISIONS)
    sources = {}
    for dataset, path in (("csqa", args.csqa), ("obqa", args.obqa), ("gsm8k", args.gsm8k)):
        rows = read_rows(path)
        sources[dataset] = dict(sha256=sha(path), original_rows=len(rows), revision=revisions[dataset], split="train")
        for i, row in enumerate(rows):
            if dataset != "gsm8k":
                built = choice_record(row, dataset)
            else:
                question, answer = row["question"], CALCULATOR.sub("", row["answer"])
                steps, marker, number = answer.rpartition("####")
                built = (dict(dataset=dataset, sample_id="gsm/" + hash_order(identity(question), "gsmid"), source_id=i,
                              question=question, source_solution=row["answer"], solution=answer, answer=number.strip())
                         if marker and steps.strip() and NUMERIC.fullmatch(number.strip()) else None)
            if built is None:
                stats[dataset]["missing_or_ambiguous"] += 1
            else:
                records.append(built)
    spec = json_read(args.science / "SCIENCEQA_SOURCE.json")
    for fname, receipt in spec["files"].items():
        if sha(args.science / fname) != receipt["sha256"]:
            raise ValueError(f"Source SHA differs: {fname}")
    problems = json_read(args.science / "problems.json")
    splits = json_read(args.science / "pid_splits.json")
    for split in ("val", "test"):
        for pid in splits[split]:
            refs.add(identity(problems[pid]["question"]), f"scienceqa/{split}/{pid}")
    sources["scienceqa"] = dict(spec, split="train", original_rows=len(splits["train"]))
    for pid in splits["train"]:
        record, reason = science_record(pid, problems[pid])
        if reason:
            stats["scienceqa"][reason] += 1
        else:
            records.append(record)
    usable = defaultdict(list)
    for record in disjoint_components(records, refs, audit):
        row = length_check(record, tok, stats[record["dataset"]])
        if row:
            usable[row["dataset"]].append(row)
    output = args.out
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for ds, n in QUOTAS.items():
        rows = select(usable[ds], n+DEV_PER_SOURCE, ds)
        if len(rows) < n+DEV_PER_SOURCE:
            raise ValueError(f"Only {len(rows)} eligible {ds}, need {n+DEV_PER_SOURCE}; {stats[ds]}")
        if ds == "gsm8k":
            dev = [row for i,row in enumerate(rows) if i % ((n+DEV_PER_SOURCE)//4) < DEV_PER_SOURCE//4]
            held = {r["sample_id"] for r in dev}
            train = [row for row in rows if row["sample_id"] not in held]
        else:
            train, dev = rows[:n], rows[n:]
        assert (len(train), len(dev)) == (n, DEV_PER_SOURCE)
        summary[ds] = dict(sources[ds], eligible=len(usable[ds]), filters=dict(stats[ds]), splits={})
        for split, subset in (("train", train), ("development", dev)):
            target = output / f"{ds}_{split}.jsonl"
            save(target, subset)
            body = dict(n=len(subset), ids=[x["source_id"] for x in subset], sha256=sha(target),
                        available_view_tokens={view: sum(row["views"][view]["response_tokens"] for row in subset) for view in subset[0]["views"]})
            summary[ds]["splits"][split] = body
        print(ds, n, len(dev), len(usable[ds]), dict(stats[ds]), flush=True)
    save(output / "EXCLUSIONS.jsonl", audit)
    manifest = dict(sources=summary, reserves=provenance, reserved_unique_questions=len(refs.exact),
                    model_tokenizer_sha256=sha(args.model / "tokenizer.json"), max_total=MAX_TOTAL, max_response=MAX_RESPONSE,
                    duplicate_rule="NFKC casefold, exact or word trigram Jaccard >= .8 for stems of six or more words, connected components across training sources",
                    exclusions_sha256=sha(output / "EXCLUSIONS.jsonl"))
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "csqa", "obqa", "gsm8k", "science", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--reserve", type=Path, action="append", default=[])
    build(parser.parse_args())
