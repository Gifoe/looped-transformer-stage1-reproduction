"""Locked factorial population for the semantic counterfactual experiment."""
from collections import Counter, defaultdict
import json
import random
import re


OPTIONS = ("apple", "bread", "cream", "grass", "honey", "lemon", "mango", "peach",
           "pearl", "plant", "river", "sheep", "stone", "sugar", "tiger", "water")
SEEDS = {"train": 2026112313, "heldout": 2026112347, "generation": 2026112389}
FAMILIES = ("lookup", "connection")
VARIANTS = ("canonical", "paraphrase", "alternate_route")


def render(family, variant, nodes, assignment, query):
    first, second = assignment
    a, b, c, d, e, f = nodes
    if family == "lookup":
        if variant == "canonical":
            return (f"Warehouse log. Carrier {a} holds {first}. Carrier {b} holds {second}. "
                    f"Which item does carrier {(a, b)[query]} hold?")
        if variant == "paraphrase":
            return (f"In the warehouse, {first} is in the care of carrier {a}, while {second} is "
                    f"in the care of carrier {b}. Name the item in the care of carrier {(a, b)[query]}.")
        return (f"Warehouse log. Badge {c} is assigned to carrier {a}. Badge {d} is assigned to carrier {b}. "
                f"Carrier {a} holds {first}. Carrier {b} holds {second}. Which item is held by the person "
                f"assigned badge {(c, d)[query]}?")
    if variant == "canonical":
        return (f"Dispatch log. Port {a} connects to hub {c}. Port {b} connects to hub {d}. "
                f"Hub {c} handles {first}. Hub {d} handles {second}. Which item is handled by the "
                f"hub connected to port {(a, b)[query]}?")
    if variant == "paraphrase":
        return (f"Port {a} sends all deliveries through hub {c}, and port {b} sends all deliveries through hub {d}. "
                f"The cargo handled at hub {c} is {first}. The cargo handled at hub {d} is {second}. "
                f"Name the cargo handled at the hub serving port {(a, b)[query]}.")
    return (f"Dispatch log. Port {a} feeds gate {e}. Port {b} feeds gate {f}. "
            f"Gate {e} connects to hub {c}. Gate {f} connects to hub {d}. "
            f"Hub {c} handles {first}. Hub {d} handles {second}. Which item is handled by the "
            f"hub reached from port {(a, b)[query]}?")


def population(split):
    if split not in ("train", "heldout"):
        raise ValueError(f"Unknown population split {split}")
    rng = random.Random(SEEDS[split])
    gap = (5, 6, 7) if split == "train" else (1, 2, 3, 4)
    schedule = [(word, OPTIONS[(i + stride) % len(OPTIONS)]) for stride in gap
                for i, word in enumerate(OPTIONS)]
    groups = rng.sample(range(10000 if split == "train" else 50000,
                              50000 if split == "train" else 90000), len(schedule) * 12)
    output = []
    for family_index, family in enumerate(FAMILIES):
        pairings = schedule[family_index::2] if split == "train" else schedule[:]
        rng.shuffle(pairings)
        query_slots = [0, 1] * (len(pairings) // 2)
        rng.shuffle(query_slots)
        for index, pair in enumerate(pairings):
            group = f"{split}:{family}:{index:03d}"
            start = 6 * (family_index * len(schedule) + index)
            nodes = tuple(f"K{number}" for number in groups[start:start + 6])
            base_order = pair if rng.randrange(2) else pair[::-1]
            query = query_slots[index]
            for variant in (VARIANTS[:1] if split == "train" else VARIANTS):
                for reversal in (0, 1):
                    facts = base_order if reversal == 0 else base_order[::-1]
                    question = render(family, variant, nodes, facts, query)
                    for remap in (0, 1):
                        options = base_order if remap == 0 else base_order[::-1]
                        labels = ("A", "B")
                        gold = facts[query]
                        slots = "\n".join(f"{label}. {word}" for label, word in zip(labels, options, strict=True))
                        prompt = f"Question: {question}\n{slots}\nAnswer with the option text.\nAnswer:"
                        output.append(dict(
                            id=f"{group}:{variant}:r{reversal}:m{remap}", group=group, family=family,
                            variant=variant, split=split, reversal=reversal, remap=remap, question=question,
                            prompt=prompt, options=dict(zip(labels, options, strict=True)),
                            choices={"label": labels, "text": options}, answerKey=labels[options.index(gold)],
                            gold_content=gold, facts=facts, query_slot=query, nodes=nodes,
                        ))
    return output


def grouped(rows):
    result = defaultdict(list)
    for row in rows:
        result[row["group"]].append(row)
    return dict(result)


def audit(train, test):
    if len(train) != 192 or len(test) != 1536:
        raise ValueError("Unexpected synthetic population size")
    all_rows = train + test
    if len({row["id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Question identities overlap")
    train_nodes = {node for row in train for node in row["nodes"]}
    test_nodes = {node for row in test for node in row["nodes"]}
    if train_nodes & test_nodes:
        raise ValueError("Synthetic Train and held-out identifiers overlap")
    train_pairs, test_pairs = (
        {frozenset(row["choices"]["text"]) for row in split} for split in (train, test)
    )
    if train_pairs & test_pairs:
        raise ValueError("Synthetic Train and held-out candidate pairs overlap")
    for group, rows in grouped(all_rows).items():
        by_cell = {(r["variant"], r["reversal"], r["remap"]): r for r in rows}
        variants = VARIANTS if rows[0]["split"] == "heldout" else VARIANTS[:1]
        if set(by_cell) != {(v, r, m) for v in variants for r in (0, 1) for m in (0, 1)}:
            raise ValueError(f"Missing matched factorial cell in {group}")
        for variant in variants:
            for remap in (0, 1):
                a, b = [by_cell[variant, flip, remap] for flip in (0, 1)]
                if a["facts"] != b["facts"][::-1] or a["options"] != b["options"]:
                    raise ValueError(f"Invalid reversal contrast in {group}")
                if (a["gold_content"] == b["gold_content"] or a["answerKey"] == b["answerKey"]
                        or Counter(re.findall(r"\w+", a["question"])) != Counter(re.findall(r"\w+", b["question"]))):
                    raise ValueError(f"Reversal did not exchange the answer and preserve words in {group}")
                for row in (a, b):
                    remapped = by_cell[variant, row["reversal"], 1 - remap]
                    if (row["question"] != remapped["question"] or row["answerKey"] == remapped["answerKey"]
                            or row["gold_content"] != remapped["gold_content"]
                            or row["choices"]["text"] != remapped["choices"]["text"][::-1]
                            or row["gold_content"] != by_cell["canonical", row["reversal"], remap]["gold_content"]):
                        raise ValueError(f"Invalid option, paraphrase or route control in {group}")
    test_counts = Counter((r["family"], r["variant"], r["reversal"], r["remap"]) for r in test)
    test_gold = Counter((r["family"], r["variant"], r["gold_content"]) for r in test)
    if set(test_counts.values()) != {64} or set(test_gold.values()) != {16}:
        raise ValueError("Held-out identities or factorial assignments are imbalanced")
    return dict(
        seed=SEEDS, train_groups=len(grouped(train)), train_rows=len(train),
        heldout_groups=len(grouped(test)), heldout_rows=len(test),
        train_candidate_pairs=len(train_pairs), heldout_candidate_pairs=len(test_pairs),
        candidate_lengths=sorted(set(map(len, OPTIONS))),
        per_factorial_cell={":".join(map(str, key)): n for key, n in sorted(test_counts.items())},
        per_variant_family_gold={":".join(key): n for key, n in sorted(test_gold.items())},
        train_evidence_identifiers=len(train_nodes), heldout_evidence_identifiers=len(test_nodes),
        heldout_family_variant_query_balance={
            ":".join(map(str, key)): n for key, n in sorted(
                Counter((r["family"], r["variant"], r["query_slot"]) for r in test).items())},
        content_templates="Reversal exchanges only the two evidence answers. Remap exchanges only option order.",
    )


def evidence_positions(tokenizer, row, prefix):
    # The synthetic evidence spelling is a single whitespace-bound answer token.
    slots = {}
    question_end = len("Question: ") + len(row["question"])
    for word in row["facts"]:
        matches = list(re.finditer(rf"(?<!\w){re.escape(word)}(?!\w)", row["prompt"][:question_end]))
        if len(matches) != 1:
            raise ValueError(f"Expected a unique fact mention of {word} in {row['id']}")
        end = matches[0].end()
        start = matches[0].start() - 1
        to_left = tokenizer.encode(row["prompt"][:start], add_special_tokens=False)
        through = tokenizer.encode(row["prompt"][:end], add_special_tokens=False)
        if len(through) != len(to_left) + 1 or prefix[:len(through)] != through:
            raise ValueError(f"Evidence token is not an anchored single token in {row['id']}")
        slots[word] = len(to_left)
    return sorted(slots.values())


def generation_subset(rows, per_family=6):
    rng = random.Random(SEEDS["generation"])
    selected = set()
    for family in FAMILIES:
        names = sorted({row["group"] for row in rows if row["family"] == family})
        selected.update(rng.sample(names, per_family))
    return [row for row in rows if row["group"] in selected and row["variant"] == "canonical"]


def write_population(out, train, heldout):
    out.mkdir(parents=True, exist_ok=True)
    manifest = audit(train, heldout)
    for split, rows in (("train", train), ("heldout", heldout)):
        save(out / f"{split.upper()}.jsonl", rows, jsonl=True)
    save(out / "DESIGN.json", manifest)


def save(path, value, jsonl=False):
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", encoding="utf-8", newline="\n") as file:
        for row in value if jsonl else (value,):
            file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)
