"""Construct grouped counterfactual worlds and pinned benchmark requests."""
import hashlib
import json
from pathlib import Path
import random
import re
from zipfile import ZipFile

TASKS = ("multihop", "prontoqa_derived", "arithmetic")
BENCHMARKS = ("prontoqa", "arc", "hellaswag")
PROOF_SOURCE = "2hop_OOD_ModusPonens_random_noadj.json"
BENCH_PROOFS = "3hop_ProofsOnly_random_noadj.json"
WORDS = ("amber", "blue", "copper", "green", "ivory", "jade", "lilac",
         "navy", "olive", "pearl", "red", "silver", "teal", "white", "yellow")
PEOPLE = ("Alex", "Blake", "Casey", "Drew", "Evan", "Jamie", "Jordan", "Morgan",
          "Quinn", "Riley", "Robin", "Sam", "Taylor", "Terry", "Zoe")
PLACES = ("Athens", "Boston", "Cairo", "Denver", "Dublin", "Hanoi", "Lima",
          "London", "Miami", "Oslo", "Paris", "Rome", "Seoul", "Tokyo", "Zurich")
CLASSES = ("lempus", "gorpus", "vumpus", "yumpus", "sterpus", "numpus",
           "tumpus", "dumpus", "rompus", "wumpus", "jompus", "shumpus", "lorpus")


def rng_for(key):
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big"))


def choice_prompt(question, choices):
    return question + "\n" + "\n".join(
        f"{chr(65 + i)}. {option}" for i, option in enumerate(choices)
    ) + "\nAnswer:"


def proof_data(path, member):
    with ZipFile(path) as archive:
        rows = json.loads(archive.read(member))
    return [(i, wrapper["test_example"]) for i, wrapper in rows.items()]


def multihop(rng, variant):
    p, q, z = rng.sample(PEOPLE, 3)
    boxes = rng.sample(WORDS, 4)
    cities = rng.sample(PLACES, 2)
    initial = rng.randrange(2)
    first_order = rng.sample(range(3), 3)
    second_order = rng.sample(range(2), 2)
    def first(a, u):
        assignments = [(p, boxes[initial ^ a]), (q, boxes[initial ^ a ^ 1]),
                       (z, boxes[2 + u])]
        return "\n".join([f"Follow the records to find the destination for {p}.",
                          *(f"{assignments[i][0]} is assigned to the {assignments[i][1]} box."
                            for i in first_order)])

    return (
        first,
        lambda b: "\n".join(
            f"The {boxes[i]} box goes to {cities[i ^ b]}." for i in second_order
        ) + f"\nWhich city is the destination for {p}?",
        lambda a, b: cities[initial ^ a ^ b],
        lambda a: boxes[initial ^ a],
        f"Which box is assigned to {p}?", None,
    )


def arithmetic(rng, variant):
    a, b, c, d = rng.sample(range(12, 49), 4)
    initial = rng.randrange(2)
    starts = (a, a + rng.choice((-5, -4, -3, 3, 4, 5)))
    middle = tuple(i + b for i in starts)
    multiplier = rng.choice((2, 3))
    bias = rng.randrange(13, 70)
    compensating = bias + multiplier * (middle[initial] - middle[initial ^ 1])

    def first(a, u):
        return (f"A separate example in the margin adds {c + 2 * u} to {d}.\n"
                f"For the requested calculation, first add {starts[initial ^ a]} and {b}.")

    def second(b):
        return (f"Multiply the result of the first step by {multiplier}, then add "
                f"{(bias, compensating)[b]}.\nWhat is the final result?")

    return (first, second,
            lambda a, b: str(multiplier * middle[initial ^ a] + (bias, compensating)[b]),
            lambda a: str(middle[initial ^ a]), "What is the result of the first step?", None)


def prontoqa_derived(rng, variant):
    source_id, proof = variant
    subject, start = re.fullmatch(r"(\w+) is an? (\w+)\.", proof["chain_of_thought"][0]).groups()
    who, bridge = re.fullmatch(r"(\w+) is an? (\w+)\.", proof["chain_of_thought"][2]).groups()
    assert who == subject and start != bridge
    alt, nuisance, nuisance_alt, nuisance3 = rng.sample(
        sorted(set(CLASSES) - {start, bridge}), 4
    )
    bridge_names = [bridge, alt]
    rng.shuffle(bridge_names)
    initial = bridge_names.index(bridge)
    person2, person3 = rng.sample([p for p in PEOPLE if p != subject], 2)
    pred = ("opaque", "metallic", "wooden", "floral", "red", "windy")
    attribute = rng.choice(pred)
    second_order = rng.sample(range(2), 2)
    premises_order = rng.sample(range(4), 4)

    def first(a, u):
        facts = [
            f"Every {start} is a {bridge_names[initial ^ a]}.",
            f"Every {nuisance} is a {bridge_names[initial ^ a ^ 1]}.",
            f"{subject} is a{'n' if start[0] in 'aeiou' else ''} {start}.",
            f"{person2} is a {nuisance}. {person3} is a {(nuisance_alt, nuisance3)[u]}.",
        ]
        return f"Decide what follows about {subject} from the premises.\n" + "\n".join(
            facts[i] for i in premises_order
        )

    def second(b):
        rules = [f"Every {item} is {'not ' if i ^ b else ''}{attribute}."
                 for i, item in enumerate(bridge_names)]
        return "\n".join(rules[i] for i in second_order) + f"\nIs {subject} {attribute}?"

    return (first, second, lambda a, b: ("Yes", "No")[initial ^ a ^ b],
            lambda a: bridge_names[initial ^ a],
            f"Which class does {subject} belong to by applying one rule?", source_id)


def controlled_group(task, i, variant=None, attempt=0):
    rng = rng_for(f"data-v2/{task}/{i}/{attempt}")
    generator = {"multihop": multihop, "arithmetic": arithmetic,
                 "prontoqa_derived": prontoqa_derived}[task]
    first, second, target, intermediate, intermediate_question, source_id = generator(rng, variant)
    group = f"{task}-{i:03d}"
    split = "pilot" if i < 8 else "development" if i < 24 else "confirmation" if i < 54 else "trajectory"
    options = sorted({target(a, b) for a in range(2) for b in range(2)})
    rng.shuffle(options)
    middle_options = sorted(intermediate(a) for a in range(2))
    rng.shuffle(middle_options)
    worlds = {}
    for name, a, b, u in (("base", 0, 0, 0), ("first", 1, 0, 0), ("second", 0, 1, 0),
                           ("same_answer", 1, 1, 0), ("irrelevant", 0, 0, 1), ("irrelevant_same", 1, 1, 1)):
        prefix = first(a, u) + "\nEnd of first stage."
        worlds[name] = {
            "prompt": choice_prompt(prefix + "\n" + second(b), options),
            "probe_prompt": choice_prompt(prefix + "\n" + intermediate_question, middle_options),
            "anchor": len(prefix), "answer": options.index(target(a, b)),
            "middle_answer": middle_options.index(intermediate(a)),
            "first_changed": bool(a), "second_changed": bool(b), "irrelevant_changed": bool(u),
        }
    assert worlds["base"]["answer"] == worlds["same_answer"]["answer"]
    assert worlds["base"]["answer"] != worlds["first"]["answer"]
    assert worlds["base"]["middle_answer"] != worlds["same_answer"]["middle_answer"]
    return {"task": task, "id": group, "split": split, "source": source_id, "generation_attempt": attempt,
            "options": options, "middle_options": middle_options, "worlds": worlds}


def fixed_sample(rows, task, count):
    indices = list(range(len(rows)))
    rng_for(f"sample-v1/{task}").shuffle(indices)
    return [rows[j] for j in indices[:count]]


def benchmarks(data):
    import pyarrow.parquet as pq

    for i, (source_id, proof) in enumerate(fixed_sample(
        proof_data(data / "prontoqa.zip", BENCH_PROOFS), "prontoqa", 90
    )):
        proposition = proof["query"].removeprefix("Prove: ").removesuffix(".")
        if " is not " in proposition:
            alt = proposition.replace(" is not ", " is ", 1)
        else:
            assert " is " in proposition
            alt = proposition.replace(" is ", " is not ", 1)
        options = [proposition, alt]
        rng_for(f"logic-options/{source_id}").shuffle(options)
        yield "prontoqa", i, source_id, proof["question"], (
            "Which statement follows?\n" + "\n".join(
                f"{chr(65+j)}. {value}" for j, value in enumerate(options)
            ) + "\nAnswer:"
        ), options, options.index(proposition), proof["chain_of_thought"]

    arc = pq.read_table(data / "arc.parquet").to_pylist()
    for i, row in enumerate(fixed_sample(arc, "arc", 100)):
        options = row["choices"]["text"]
        yield "arc", i, row["id"], "", choice_prompt(
            "Question: " + row["question"], options
        ), options, row["choices"]["label"].index(row["answerKey"]), None

    hella = pq.read_table(data / "hellaswag.parquet").to_pylist()
    for i, row in enumerate(fixed_sample(hella, "hellaswag", 50)):
        context = row["ctx"]
        options = row["endings"]
        yield "hellaswag", i, str(row["ind"]), context, choice_prompt(
            "Choose the most likely continuation.\n" + context, options
        ), options, int(row["label"]), None


def aligned_group(task, i, variant, accept):
    for attempt in range(100):
        group = controlled_group(task, i, variant, attempt)
        if accept(group):
            return group
    raise RuntimeError(f"No token-aligned realization for {task}/{i}")


def build(data: Path, accept):
    source = proof_data(data / "prontoqa.zip", PROOF_SOURCE)
    source = [entry for entry in source
              if len(entry[1]["chain_of_thought"]) == 5
              and all(re.fullmatch(r"\w+ is an? \w+\.", entry[1]["chain_of_thought"][j]) for j in (0, 2))
              and entry[1]["chain_of_thought"][0].split()[-1] != entry[1]["chain_of_thought"][2].split()[-1]]
    variants = fixed_sample(source, "prontoqa_derived", 60)
    if len(variants) < 54:
        raise RuntimeError(f"Too few eligible PrOntoQA proof templates: {len(variants)}")
    groups = [aligned_group(task, i, variant, accept)
              for task, entries in (("multihop", [None] * 80), ("arithmetic", [None] * 80),
                                     ("prontoqa_derived", variants))
              for i, variant in enumerate(entries)]
    for task, i, source_id, anchor, prompt, options, answer, proof in benchmarks(data):
        groups.append({"task": task, "id": f"{task}-{i:03d}", "split": "pilot" if i < 8 else "trajectory",
                       "source": source_id, "options": options, "proof": proof,
                       "worlds": {"base": {"prompt": anchor + "\n" + prompt if task == "prontoqa" else prompt,
                                           "anchor": (len(anchor) if task == "prontoqa" else len("Choose the most likely continuation.\n" + anchor) if anchor else None), "answer": answer}}})
    return groups



