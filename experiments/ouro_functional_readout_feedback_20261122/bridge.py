"""Load the frozen ARC model and the previous release without copying its code."""
import json
from pathlib import Path
import sys


def configure(previous, archive_path):
    sys.path.insert(0,str(Path(previous).resolve()))
    import arc_features
    import amplify
    old=arc_features.archive(Path(archive_path))
    return old,arc_features,amplify


def rows(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def winner(score, row, norm):
    return max(range(len(score)), key=lambda i:score[i] / (row['characters'][i] if norm else 1))


def dump(path, obj):
    path=Path(path)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)
