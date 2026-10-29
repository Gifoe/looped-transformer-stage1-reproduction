"""Question-only index for future split audits; no new benchmark is evaluated."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'frozen_loop_update_value_seed2021_v1'
HISTORY=ROOT.parent/'looped_stage1_trajectory_audit_seed2021_v1'
manifest=json.loads((OLD/'data/manifest_audit.json').read_text())
paths=[HISTORY/x['path'] for x in manifest['exposure_scan']]
paths += [OLD/'data'/f'{name}.jsonl' for name in manifest['manifests']]
questions=set();ids=set();sources=[]
for p in paths:
    count=0
    with p.open(encoding='utf8') as f:
        for line in f:
            if not line.strip():continue
            row=json.loads(line);q=row.get('question')
            if not isinstance(q,str):continue
            # Do not read gold or candidates. Normalized equality is conservative,
            # but does not detect paraphrases or unknown prior exposures.
            normalized=' '.join(q.casefold().split())
            questions.add(hashlib.sha256(normalized.encode('utf8')).hexdigest())
            if isinstance(row.get('id'),str):ids.add(row['id'])
            count+=1
    sources.append(dict(path=str(p.relative_to(ROOT.parent)).replace('\\','/'),question_rows=count,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
record=dict(scope='Known historical manifests plus all first-round manifests; not a semantic-overlap or pretraining-contamination guarantee',normalized_question_hashes=sorted(questions),known_ids=sorted(ids),sources=sources,unique_questions=len(questions),unique_ids=len(ids),new_dataset_examples_read=False)
(ROOT/'results/KNOWN_EXPOSURE_INDEX.json').write_text(json.dumps(record,indent=2),encoding='utf8')
print(json.dumps({k:record[k] for k in ['unique_questions','unique_ids','new_dataset_examples_read']}))
