#!/usr/bin/env bash
set -euo pipefail
cd /root/ouro-native-state-20261121
h=/root/autodl-tmp/ouro-cross-task-20261118
export OPENBLAS_NUM_THREADS=4
mkdir -p logs
"$h/venv/bin/python" - <<'PY'
from pathlib import Path
from data_protocol import multiple_choice_rows, choose, serialize, gold_free
from native import save
p=Path('/root/autodl-tmp/ouro-arc-pareto-20261119/data/train.parquet')
rows=multiple_choice_rows(p,'arc_challenge')
ordered,_=choose(rows,'arc',set(),set(),800)
caps={'pilot':ordered[:24],'dev':ordered[24:120],'confirm_arc':ordered[120:216]}
save('ARC_BLIND_INPUTS.json',{k:[{x:y for x,y in serialize(v,0).items() if x!='gold'} for v in g] for k,g in caps.items()})
PY
test -s results/RUNTIME.json
"$h/venv/bin/python" -u observe.py "$h/model" results ARC_BLIND_INPUTS.json pilot > logs/atlas.pilot.log 2>&1
"$h/venv/bin/python" -u observe.py "$h/model" results ARC_BLIND_INPUTS.json dev > logs/atlas.dev.log 2>&1
"$h/venv/bin/python" -u observe.py "$h/model" results ARC_BLIND_INPUTS.json freeze > logs/freeze.log 2>&1
printf 'DONE\n'

