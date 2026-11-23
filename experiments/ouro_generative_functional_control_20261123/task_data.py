"""Build deterministic answer prompts and normalize source records."""

import hashlib
import json
from pathlib import Path
import unicodedata
import re
import random


SEED=261123137
LETTER="Answer with the option letter."
CONTENT="Answer with the full text of the correct option."


def hash_order(key,salt):
    return hashlib.sha256(f"{SEED}/{salt}/{key}".encode()).hexdigest()


def load(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save(path,rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".partial")
    with tmp.open("w",encoding="utf-8",newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
    tmp.replace(path)


def identity(x):
    return " ".join(re.findall(r"\w+",unicodedata.normalize("NFKC",x).casefold()))


def multiple_choice(row,dataset,kind=None,shuffle=False):
    original=list(range(len(row["choices"]["label"])))
    ix=row["choices"]["label"].index(row["answerKey"])
    if shuffle:
        random.Random(int(hash_order(row["id"],"shuffle"),16)).shuffle(original)
    options=[row["choices"]["text"][j] for j in original]
    labels=[chr(65+i) for i in range(len(options))]
    pick=original.index(ix)
    tag=kind or ("letter" if int(hash_order(row["id"],"format"),16)%10<3 else "content")
    prompt=mc_prompt(row["question"],options,tag)
    response=labels[pick] if tag=="letter" else options[pick]
    return dict(instruction=LETTER if tag=="letter" else CONTENT,context=row["question"],
                response=response,task_type=f"mc_{tag}",dataset=dataset,sample_id=row["id"],
                prompt=prompt, options=options,label=labels[pick],answer_content=options[pick],
                candidate=dict(id=row["id"],question=row["question"],choices=dict(label=labels,text=options),
                               answerKey=labels[pick]))


def mc_prompt(question,options,tag):
    labels=[chr(65+i) for i in range(len(options))]
    parts="\n".join(f"{x}. {t}" for x,t in zip(labels,options,strict=True))
    return f"Question: {question}\n{parts}\n{LETTER if tag=='letter' else CONTENT}\nAnswer:"


def encode_target(tok,record):
    prompt=record["prompt"]
    prefix=tok.encode(prompt,add_special_tokens=False)
    whole=tok.encode(prompt+" "+record["response"],add_special_tokens=False)
    assert whole[:len(prefix)]==prefix and len(whole)>len(prefix), record["sample_id"]
    eos=tok.eos_token_id
    if eos is None:
        raise ValueError("Tokenizer has no EOS")
    return prefix,whole+[eos],len(whole)-len(prefix)+1
