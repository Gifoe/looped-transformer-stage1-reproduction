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

out=Path('t2_full/results')
protocol=Path('mixed_full/results/FREEZE.json')
frozen=json.loads(protocol.read_text())
t1_protocol=Path('t1_full/results/FREEZE.json')
assert sha(t1_protocol)=='87743067b1b009774132c327895f76f7e9409b22e239daf9383c8cecc4fe2317'
assert sha(protocol)=='2d966f1d2cfd37f7c9447af85bce49a604cbea2ef5266b9d22ae20165fef1383'
weight=Path(os.environ.get('OURO_T2_WEIGHT',str(Path(os.environ['controllability'])/'results/weights/rank1_t2.pt')))
digest=sha(weight)
assert digest=='117f5f2fe9c5afe363ba8c032f42cd25155f7a88cb465e4e79bfb3bf496c8032'
lock=dict(frozen,selection='Historical Rank-1 T2, fixed before scoring',
          arms={'t2':dict(path=str(weight.resolve()),sha256=digest)},
          entry_sha256=sha('t2_full/t2_evaluate.py'),mixed_protocol_sha256=sha(protocol),t1_protocol_sha256=sha(t1_protocol))
path=out/'FREEZE.json'
if path.exists():
    assert json.loads(path.read_text())==lock
else:
    tmp=path.with_suffix('.partial')
    tmp.write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
    tmp.replace(path)
print('Locked T2',digest,'at K4 for',sum(lock['expected'].values()),'questions',flush=True)
PY
out=t2_full/results
args=(--archive "$archive" --previous "$previous" --model "$model" --minimal "$feedback" --stage2 mixed_full/archive       --unified "$unified" --followup "$feedback/generation_followup" --historical "$historical"       --out "$out/predictions" --freeze "$out/FREEZE.json" --arm t2)
for task in sciq piqa arc_challenge arc_easy winogrande mmlu hellaswag mmlu_pro; do
    printf '%s START %s\n' "$(date -Is)" "$task"
    bash gpu_job.sh "t2_full_$task" t2_full/t2_evaluate.py "${args[@]}" --task "$task" >> "t2_full/logs/$task.log" 2>&1
    printf '%s COMPLETE %s\n' "$(date -Is)" "$task"
done
printf 'DONE\n'
