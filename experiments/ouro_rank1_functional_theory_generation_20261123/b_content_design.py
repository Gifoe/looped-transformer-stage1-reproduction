"""Paired two-hop problems for the content-specific Rank-1 experiment."""

from collections import Counter
from itertools import combinations
import random
import re

WORDS = ("apple", "bread", "cream", "grass", "honey", "lemon", "mango", "peach",
         "pearl", "plant", "river", "sheep", "stone", "sugar", "tiger", "water")
CODES = tuple(f"{word} {WORDS[(i + gap) % len(WORDS)]}"
              for gap in (5, 7) for i, word in enumerate(WORDS))
FAMILIES = ("freight", "travel")
SEEDS = {"dev": 20261123801, "heldout": 20261123813, "lexical_pairings": 20261123851,
         "gen_selection": 20261123893}
SIZES = {"dev": 24, "heldout": 128}


def render(family, first, second, link_order, fact_order, target, placebo, values):
    if family == "freight":
        intro = ["Freight directory.", "Each depot points to one bay. Each bay has a two-word code."]
        links = [f"Depot {first[j]} points to Bay {second[j]}." for j in link_order]
        lines = {j: f"Bay {second[j]} code: " for j in fact_order}
        query = f"Which code is recorded at the bay that Depot {first[target]} points to?"
    else:
        intro = ["Travel directory.", "Each traveler takes one shuttle. Each shuttle has a two-word code."]
        links = [f"Traveler {first[j]} takes Shuttle {second[j]}." for j in link_order]
        lines = {j: f"Shuttle {second[j]} code: " for j in fact_order}
        query = f"Which code is recorded for the shuttle that Traveler {first[target]} takes?"
    text = "\n".join([*intro, *links]) + "\n"
    spans = {}
    for j in fact_order:
        text += lines[j]
        start = len(text)
        text += values[j] + "."
        spans[j] = (start, len(text))
        text += "\n"
    text += f"Question: {query}\nGive only the two-word code.\nAnswer:"
    return text, [spans[target], spans[placebo]]


def population():
    rng = random.Random(SEEDS["lexical_pairings"])
    pairs = [pair for pair in combinations(CODES, 2) if not set(pair[0].split()) & set(pair[1].split())]
    rng.shuffle(pairs)
    rows = {}
    offset = 0
    for split, count in SIZES.items():
        rng = random.Random(SEEDS[split])
        start = 1000 if split == "dev" else 5000
        names = rng.sample(range(start, start + 4000), count * 8)
        output = []
        for i, options in enumerate(pairs[offset:offset + count]):
            family = FAMILIES[i % len(FAMILIES)]
            code_order = [0, 0, 1, 1]
            rng.shuffle(code_order)
            target = rng.randrange(4)
            placebo = next(j for j, slot in enumerate(code_order) if j != target and slot == code_order[target])
            pos = names[i * 8:i * 8 + 8]
            first, second = (tuple(f"R{x}" for x in pos[:4]), tuple(f"J{x}" for x in pos[4:]))
            links = rng.sample(range(4), 4)
            facts = rng.sample(range(4), 4)
            identity = f"{split}:{family}:{i:03d}"
            worlds = []
            for flip in (0, 1):
                assignment = [options[slot ^ flip] for slot in code_order]
                text, spans = render(family, first, second, links, facts, target, placebo, assignment)
                worlds.append(dict(id=f"{identity}:r{flip}", group=identity, family=family, split=split,
                                   prompt=text, options=list(options), code_index=code_order[target] ^ flip,
                                   source_spans=spans, query_id=first[target], target_id=second[target],
                                   control_id=second[placebo], entity_ids=[*first, *second],
                                   link_order=links, fact_order=facts, flip=flip))
            output.append(worlds)
        rows[split] = output
        offset += count
    audit(rows)
    return rows


def audit(rows):
    entities, lexical_pairs = {}, {}
    for split, pairs in rows.items():
        if len(pairs) != SIZES[split]:
            raise RuntimeError(f"Unexpected pair count for {split}")
        entities[split] = {name for pair in pairs for name in pair[0]["entity_ids"]}
        lexical_pairs[split] = {tuple(pair[0]["options"]) for pair in pairs}
        if tuple(Counter(pair[0]["family"] for pair in pairs).values()) != (len(pairs) // 2,) * 2:
            raise RuntimeError("Unbalanced family counts")
        for a, b in pairs:
            if (a["code_index"] == b["code_index"] or a["source_spans"] == [] or
                    a["options"] != b["options"] or a["query_id"] != b["query_id"] or
                    Counter(re.findall(r"\w+", a["prompt"])) != Counter(re.findall(r"\w+", b["prompt"]))):
                raise RuntimeError("Fact reversal changed something beyond the two paired content assignments")
            for row in (a, b):
                gold = row["options"][row["code_index"]]
                for lo, hi in row["source_spans"]:
                    if row["prompt"][lo:hi] != gold + ".":
                        raise RuntimeError("Target and irrelevant source spans must carry the same lexical content")
    if entities["dev"] & entities["heldout"] or lexical_pairs["dev"] & lexical_pairs["heldout"]:
        raise RuntimeError("Development and holdout identities and code pairings must be disjoint")


def encode(tokenizer, rows):
    encoded = []
    for a, b in rows:
        group = []
        for row in (a, b):
            text = row["prompt"]
            tok = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            prefix = list(tok["input_ids"])
            positions = [[i for i, (start, end) in enumerate(tok["offset_mapping"])
                          if start < hi and end > lo] for lo, hi in row["source_spans"]]
            candidates = []
            for option in row["options"]:
                ids = tokenizer.encode(text + " " + option, add_special_tokens=False)
                if ids[:len(prefix)] != prefix or len(ids) - len(prefix) != 2:
                    raise RuntimeError(f"Two answer tokens with stable boundary required for {row['id']}")
                candidates.append(ids[len(prefix):])
            if any(len(site) != 3 or max(site) >= len(prefix) - 1 for site in positions):
                raise RuntimeError(f"Two answer words and punctuation required at each source for {row['id']}")
            group.append(dict(**row, prefix=prefix, candidates=candidates, source_positions=positions))
        if [len(row["prefix"]) for row in group] != [len(group[0]["prefix"])] * 2:
            raise RuntimeError("Matched prompts must have equal model sequence lengths")
        if group[0]["source_positions"] != group[1]["source_positions"]:
            raise RuntimeError("Matched source token coordinates shifted under reversal")
        if group[0]["candidates"] != group[1]["candidates"]:
            raise RuntimeError("Candidate tokenization changed under reversal")
        encoded.append(group)
    return encoded


def generations(heldout, per_family=16):
    rng = random.Random(SEEDS["gen_selection"])
    chosen = set()
    for family in FAMILIES:
        groups = [pair[0]["group"] for pair in heldout if pair[0]["family"] == family]
        chosen.update(rng.sample(groups, per_family))
    return [pair for pair in heldout if pair[0]["group"] in chosen]
