"""Frozen 3×3 tableaux and shared-prefix counterfactual forks."""
import hashlib
import itertools
import json
import random

SPLITS = {"development": 24, "new_entities": 48, "new_structure": 48}
NAMES = (
    "Alice Bob Carol David Emily Frank Grace Henry Irene James Laura Martin Nancy Oscar Peter Rachel".split(),
    "Aaron Brian Clara Dylan Ethan Fiona Harry Isaac Jacob Kelly Louis Molly Nora Owen Paula Simon".split(),
    "Adam Bella Chloe Derek Elena Felix Heidi Jonah Kevin Lydia Mason Nina Oliver Penny Ruby Victor".split(),
)
ANSWERS = (
    "apple ball book box bread cake chair clock coat cup door drum fish flag flower glass hat key lamp milk pen phone ring shoe spoon table train tree watch".split(),
    "ant bear bird boat brush bus camera candle card cat coin desk duck goat horse map mouse pear piano plate rabbit radio rock sheep shirt snake star sword tiger".split(),
    "arm bed bell bike bowl button car carpet cheese cloud comb dog egg fork frog grape guitar hand leaf lock moon needle rabbit scarf shark shell sock window wolf".split(),
)
KEYS = (
    "red blue green yellow black white pink brown purple orange silver gold".split(),
    "north south east west left right front back upper lower middle central".split(),
    "circle square triangle diamond star heart cross oval dot line ring cube".split(),
)
FORKS = ("open", "content", "letter", "distractor", "order", "relabel", "evidence")
WORLDS = list(itertools.product(range(3), repeat=2))


def seed(*parts):
    text = "/".join(map(str, ("ouro-hidden-mechanism-v1", *parts)))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def dump(obj):
    return json.dumps(obj, ensure_ascii=False, allow_nan=False, sort_keys=True)


def load(path):
    with path.open(encoding="utf-8") as file:
        yield from map(json.loads, file)


def group(split, index, attempt):
    i = list(SPLITS).index(split)
    rng = random.Random(seed(split, index, attempt))
    name, witness = rng.sample(NAMES[i], 2)
    keys, answers = rng.sample(KEYS[i], 6), rng.sample(ANSWERS[i], 7)
    order, labels = list(range(5)), list("ABCDE")
    rng.shuffle(order)
    rng.shuffle(labels)
    return {
        "id": f"{split}-{index:03}", "split": split, "attempt": attempt, "structure": i == 2,
        "name": name, "witness": witness, "keys": keys, "answers": answers, "order": order,
        "new_labels": labels, "worlds": [{"s": s, "m": m, "y": (s + m) % 3} for s, m in WORLDS],
    }


def render(g, s, m, fork):
    prompt, spans = "", {}
    def add(text, role=None):
        nonlocal prompt
        a = len(prompt)
        prompt += text
        if role:
            spans.setdefault(role, []).append((a, len(prompt)))
    add(f"Read the facts and answer the question.\n{g['witness']} uses the")
    add(f" {g['keys'][5]}", "noise")
    add(f" token.\n{g['name']} uses the")
    add(f" {g['keys'][s]}", "source")
    add(" token.\n")
    if g["structure"]:
        for key in range(3):
            add(f"The {g['keys'][key]} token is linked to the {g['keys'][key+3]} sign.\n", "bridge")
        add("Records for signs:\n")
        for key in range(3):
            add(f"The {g['keys'][key+3]} sign identifies the {g['answers'][(key+m)%3]}.\n", "mapping")
    else:
        add("Records for tokens:\n")
        for key in range(3):
            add(f"The {g['keys'][key]} token identifies the {g['answers'][(key+m)%3]}.\n", "mapping")
    add(f"Question: Which object is identified by the token {g['name']} uses?", "query")
    prefix = prompt
    order, labels = g["order"], list("ABCDE")
    options = list(g["answers"][:5])
    if fork == "evidence":
        add(f"\n{g['witness']} and {g['name']} are in the same room.", "evidence")
    if fork == "distractor":
        options[-2:] = g["answers"][5:]
    if fork == "order":
        order = order[1:] + order[:1]
        # Candidate–label binding stays constant in the order-only branch.
        labels = labels[1:] + labels[:1]
    if fork == "relabel":
        labels = g["new_labels"]
    if fork != "open":
        for label, idx in zip(labels, order):
            add("\n")
            add(label, "label")
            add(")")
            add(f" {options[idx]}", "candidate")
        add("\nReply with the option letter." if fork == "letter" else "\nReply with the object name.")
    add("\nAnswer:", "readout")
    by_content = {j: labels[i] for i, j in enumerate(order)}
    return prompt, spans, prefix, by_content, options


def encode(tokenizer, g, s, m, fork):
    prompt, spans, prefix, by_content, options = render(g, s, m, fork)
    obj = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = obj["input_ids"]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    if ids[:len(prefix_ids)] != prefix_ids:
        raise ValueError("Shared prefix retokenizes")
    roles = {}
    for role, intervals in spans.items():
        roles[role] = sorted({i for left, right in intervals for i, (a, b) in enumerate(obj["offset_mapping"])
                              if a < right and b > left})
    roles["readout"] = [len(ids)-1]
    if len(roles["source"]) != 1 or len(roles["noise"]) != 1:
        raise ValueError("Need exact single-token source and noise interventions")
    def successors(targets):
        values = [tokenizer.encode(prompt + " " + v, add_special_tokens=False) for v in targets]
        if any(v[:-1] != ids for v in values):
            raise ValueError("A continuation is not one aligned token")
        return [v[-1] for v in values]
    contents = successors(options)
    labels = successors([by_content[i] for i in range(5)])
    return {"s": s, "m": m, "y": (s+m)%3, "fork": fork, "ids": ids, "roles": roles,
            "prefix": len(prefix_ids), "content": contents, "letter": labels, "options": options,
            "by_content": by_content, "read": len(ids)-1, "text": prompt}


def tableau(tokenizer, g, forks=("open", "content", "letter")):
    result = {fork: [encode(tokenizer, g, s, m, fork) for s, m in WORLDS] for fork in forks}
    for worlds in result.values():
        base = worlds[0]
        for e in worlds:
            if e["roles"] != base["roles"] or len(e["ids"]) != len(base["ids"]):
                raise ValueError("A tableau has role or token-count drift")
            if [i for i,(a,b) in enumerate(zip(base["ids"],e["ids"])) if a!=b and i not in
                    set(base["roles"]["source"]+base["roles"]["mapping"])]:
                raise ValueError("A tableau changed another token")
    return result


def create(tokenizer):
    for split, count in SPLITS.items():
        for index in range(count):
            for attempt in range(400):
                g = group(split, index, attempt)
                if g["new_labels"] == list("ABCDE"):
                    continue
                try:
                    tableau(tokenizer,g)
                    for fork in FORKS:
                        encode(tokenizer,g,0,0,fork)
                except ValueError:
                    continue
                yield g
                break
            else:
                raise RuntimeError(f"No valid group for {split}:{index}")
