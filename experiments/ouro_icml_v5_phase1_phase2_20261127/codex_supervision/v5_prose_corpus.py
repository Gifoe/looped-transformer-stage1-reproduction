"""Freeze the label-blind prose corpus and document-disjoint diagnostic cohorts."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import re

from v4_common import EXT_DEV_DIR, UNLABELED_AUDIT, UNLABELED_FIT, load_jsonl, save_json, sha256_file
from p1_gpu1_csbp_pure_prose import (
    CAUSAL_CUES, DISCOURSE_STARTS, content_words, detokenize_wikitext,
)

ROOT = Path("/root/autodl-tmp/ouro-v5-icml-phase1-20261127")
PRE = ROOT / "track_a_csbp/preflight"
SOURCE = UNLABELED_FIT.parents[1] / "prepared_sources"
OFFICIAL = {"arc_challenge": 1172, "arc_easy": 2376, "sciq": 1000, "mmlu": 14042,
            "hellaswag": 10042, "winogrande": 1267, "piqa": 1838, "mmlu_pro": 12032}
TEXT_FIELDS = ("question", "context", "passage", "stem", "text", "choices", "options", "query",
               "input", "prompt", "ctx", "ctx_a", "ctx_b", "activity_label", "sentence",
               "answerA", "answerB", "option1", "option2", "sol1", "sol2")
FORBIDDEN_QA = re.compile(r"(?im)^\s*(?:question\s*[:\d]|(?:answer|example)\s*:|[A-D][.)]\s|[-*•]\s|\d{1,2}[.)]\s)")


def normalize(text):
    return tuple(re.findall(r"[a-z0-9]+", detokenize_wikitext(text).lower()))


def fingerprint(values):
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def fragments(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, list):
        for value in node:
            yield from fragments(value)
    elif isinstance(node, dict):
        if "label" in node and "text" in node:
            yield from fragments(node["text"])
        else:
            for key in TEXT_FIELDS:
                if key in node:
                    yield from fragments(node[key])


def reference_index():
    exact, window, evidence = set(), set(), {}
    paths = [(f"official/{name}", SOURCE / f"blind_test_{name}.jsonl", n) for name, n in OFFICIAL.items()]
    paths += [(f"developer/{path.stem}", path, None) for path in sorted(EXT_DEV_DIR.glob("*.jsonl"))]
    if len(paths) != 8 + 7:
        raise ValueError("The blind eight benchmarks and all seven development files are required")
    for name, path, count in paths:
        rows = load_jsonl(path)
        if count is not None and (len(rows) != count or any(set(row) & {"gold", "label", "target", "answer_index", "answerKey"} for row in rows)):
            raise ValueError(f"The official reference is not complete and blinded: {path.name}")
        texts = 0
        for row in rows:
            vals = list(fragments(row))
            vals += list(fragments(row.get("candidate", {})))
            for request in row.get("requests", []):
                vals += list(fragments(request.get("args")))
            for value in vals:
                tokens = normalize(value)
                if tokens:
                    texts += 1
                    exact.add(tokens)
                    window.update(tokens[i:i+8] for i in range(max(0, len(tokens)-7)))
        if count is not None and texts < count:
            raise ValueError(f"Missing blinded text in {path.name}")
        evidence[name] = dict(n=len(rows), text_fields=texts, sha256=sha256_file(path))
    return exact, window, evidence


def collect_text(rows, exact, window, excluded_doc, excluded_windows, seen_documents=None):
    result = {"A": [], "B": []}
    counts = Counter()
    seen = set()
    for row in rows:
        doc = row["document_id"]
        if doc in excluded_doc or (seen_documents is not None and doc in seen_documents):
            counts["held_out_or_duplicate_document"] += 1
            continue
        txt = detokenize_wikitext(row["text"])
        if FORBIDDEN_QA.search(txt):
            counts["question_list_or_answer_format"] += 1
            continue
        tokens = normalize(txt)
        spans = {tokens[i:i+8] for i in range(max(0, len(tokens)-7))}
        if tokens in exact or spans & window:
            counts["benchmark_or_dev_exact_or_8gram"] += 1
            continue
        if spans & excluded_windows:
            counts["heldout_audit_8gram"] += 1
            continue
        counts["clean_document"] += 1
        for paragraph in re.split(r"(?m)\n\s*=.*=\s*\n|\n\s*\n", txt):
            sentences = re.split(r"(?<=[.!?])\s+", paragraph.strip())
            for i, sentence in enumerate(sentences[1:-1], start=1):
                sentence = re.sub(r"\s+", " ", sentence).strip()
                words = sentence.split()
                if not 16 <= len(words) <= 40 or not sentence[0].isupper() or not sentence.endswith("."):
                    continue
                if re.search(r'["“”=\[\]@?•]', sentence) or len(re.findall(r"\b[12]\d{3}\b", sentence)) >= 2:
                    continue
                low = sentence.lower()
                cue_a = any(re.search(r"\b" + re.escape(cue) + r"\b", low) for cue in CAUSAL_CUES)
                cue_b = any(low.startswith(cue.lower()) for cue in DISCOURSE_STARTS) or (cue_a and i > 0)
                if not (cue_a or cue_b):
                    continue
                boundaries = [j for j in range(max(1, (len(words)*40+99)//100), len(words)*65//100+1)
                              if 5 <= len(words)-j <= 16]
                if not boundaries:
                    continue
                pivots = re.compile(r"^(?:because|causes?|converts?|results?|leads?|increases?|decreases?|is|are|was|were|be|consists?|allows?|produces?|prevents?|absorbs?|requires?|by|in|of|to|as|from|through|using|with|into|than|that|due|for|can|may|will|would|depends?)$", re.I)
                valid = [j for j in boundaries if pivots.fullmatch(words[j-1].strip(",:;"))]
                if not valid:
                    continue
                j = min(valid, key=lambda j: abs(j/len(words)-.54))
                lead, c0, stub = " ".join(words[:j]), " ".join(words[j:]), " ".join(words[max(0, j-4):j])
                key = hashlib.sha256(f"{doc}\n{sentence}".encode()).hexdigest()[:24]
                if key in seen or normalize(c0) in exact:
                    continue
                seen.add(key)
                base = {"id": key, "doc_id": doc, "lead": lead, "c0": c0,
                        "stub": stub[0].upper() + stub[1:]}
                if cue_a:
                    result["A"].append({**base, "id": "A/"+key, "context": "", "route": "A"})
                if cue_b and i > 0:
                    # Preserve contiguous original context and reject late cuts that break the context budget.
                    context = " ".join(" ".join(s.split()) for s in sentences[max(0, i-2):i]).strip()
                    if not 14 <= len(context.split()) <= 80 or FORBIDDEN_QA.search(context):
                        continue
                    tailwords = content_words(c0)
                    copy = len(tailwords & content_words(context)) / max(1, len(tailwords))
                    if copy <= .5:
                        result["B"].append({**base, "id": "B/"+key, "context": context,
                                            "route": "B", "content_copy": copy})
    return result, counts


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def main():
    exact, window, references = reference_index()
    fit = load_jsonl(UNLABELED_FIT)
    audit = load_jsonl(UNLABELED_AUDIT)
    audit_docs = {row["document_id"] for row in audit}
    audit_aliases = {name.split("/", 2)[-1].casefold() for name in audit_docs}
    audit_windows = set()
    for row in audit:
        tokens = normalize(row["text"])
        audit_windows.update(tokens[i:i+8] for i in range(max(0, len(tokens)-7)))
    supplement = SOURCE / "unlabeled_corpus_candidates.jsonl"
    additions = load_jsonl(supplement)
    avoid = audit_docs | {row["document_id"] for row in fit+additions
                         if row["document_id"].split("/", 2)[-1].casefold() in audit_aliases}
    first, counts = collect_text(fit, exact, window, avoid, audit_windows)
    eligible = first
    extended = {}
    if min(map(len, first.values())) < 1200:
        extra, extended = collect_text(additions, exact, window, avoid, audit_windows,
                                       {row["document_id"] for row in fit})
        eligible = {route: first[route] + extra[route] for route in first}
    dedup = {}
    for route, rows in eligible.items():
        seen = set()
        dedup[route] = []
        for row in rows:
            identity = normalize(row["lead"]+" "+row["c0"])
            if identity not in seen:
                seen.add(identity)
                dedup[route].append(row)
    sources = {str(path): sha256_file(path) for path in (UNLABELED_FIT, UNLABELED_AUDIT, supplement)}
    written = {}
    for route, rows in dedup.items():
        for row in rows:
            full = normalize(f"{row['context']} {row['lead']} {row['c0']}")
            if full in exact or any(full[i:i+8] in window for i in range(max(0,len(full)-7))):
                raise ValueError("An eligible probe overlaps an evaluation reference")
        path = PRE / f"DIAGNOSTIC_ONLY_{route}.jsonl"
        save_jsonl(path, rows)
        written[route] = dict(n=len(rows), documents=len({r["doc_id"] for r in rows}),
                              sha256=sha256_file(path))
    info = dict(official_examples=sum(OFFICIAL.values()), references=references, source_sha256=sources,
                original_fit_route_counts={route:len(rows) for route, rows in first.items()},
                original_fit_audit=dict(counts), supplement_audit=dict(extended), routes=written,
                exact_and_8gram_retained_overlap=0, label_blind_reference_only=True)
    save_json(PRE / "CORPUS_DEDUP_AUDIT.json", info)
    print("V5_STRICT_CORPUS_READY", info["original_fit_route_counts"], written, dict(extended), flush=True)


if __name__ == "__main__":
    main()

