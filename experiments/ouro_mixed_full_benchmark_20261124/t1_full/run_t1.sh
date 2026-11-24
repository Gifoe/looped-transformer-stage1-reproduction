#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source external_env.sh
export controllability
"$py" - <<'PY'
from hashlib import file_digest
import json,os
from pathlib import Path

def sha(path):
    with Path(path).open('rb') as f:
        return file_digest(f,'sha256').hexdigest()

out=Path('t1_full/results')
protocol=Path('mixed_full/results/FREEZE.json')
frozen=json.loads(protocol.read_text())
assert sha(protocol)=='2d966f1d2cfd37f7c9447af85bce49a604cbea2ef5266b9d22ae20165fef1383'
weight=Path(os.environ.get('OURO_T1_WEIGHT',str(Path(os.environ['controllability'])/'results/weights/rank1_t1.pt')))
digest=sha(weight)
assert digest=='91d4919308bf9060ce0c44e4046e4851e7e59f65b9692c77045535f96534efd2'
lock=dict(frozen,selection='Historical Rank-1 T1, fixed before scoring',
          arms={'t1':dict(path=str(weight.resolve()),sha256=digest)},
          entry_sha256=sha('t1_full/t1_evaluate.py'),mixed_protocol_sha256=sha(protocol))
path=out/'FREEZE.json'
if path.exists():
    assert json.loads(path.read_text())==lock
else:
    tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)
print('Locked T1',digest,'at K4 for',sum(lock['expected'].values()),'questions',flush=True)
PY
out=t1_full/results
args=(--archive "$archive" --previous "$previous" --model "$model" --minimal "$feedback" --stage2 mixed_full/archive       --unified "$unified" --followup "$feedback/generation_followup" --historical "$historical"       --out "$out/predictions" --freeze "$out/FREEZE.json" --arm t1)
for task in sciq piqa arc_challenge arc_easy winogrande mmlu hellaswag mmlu_pro; do
    printf '%s START %s\n' "$(date -Is)" "$task"
    bash gpu_job.sh "t1_full_$task" t1_full/t1_evaluate.py "${args[@]}" --task "$task" >> "t1_full/logs/$task.log" 2>&1
    printf '%s COMPLETE %s\n' "$(date -Is)" "$task"
done
printf 'DONE\n'
