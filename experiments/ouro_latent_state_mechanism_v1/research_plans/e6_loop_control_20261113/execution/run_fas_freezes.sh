set -eu
R=/root/autodl-tmp/ouro-e6a
while [ ! -f "$R/full_answer/validation/COMPLETE.json" ]; do sleep 5; done
for item in 'fas_char5090 scores' 'fas_sum5090 raw_sum' 'fas_token5090 mean_token'; do
  set -- $item
  "$R/venv/bin/python" "$R/E6A/evaluate_capture_v2.py" --capture "$R/full_answer" --labels /root/autodl-tmp/ouro-e4-e5/inputs/labels.json --out "$R/E6A/$1" --aggregation "$2" --phase dev --transfer-spec "$R/E6A/joint5090/E6A_FROZEN_METHOD_SPEC.json"
done
"$R/venv/bin/python" - <<'PY'
import json,hashlib,datetime
from pathlib import Path
r=Path('/root/autodl-tmp/ouro-e6a/E6A')
files=[r/n/'E6A_FROZEN_METHOD_SPEC.json' for n in ['fas_char5090','fas_sum5090','fas_token5090']]
assert not list((r.parent/'full_answer/test').glob('*.json'))
(r/'FAS_DEVELOPMENT_FROZEN.json').write_text(json.dumps({'status':'FROZEN_BEFORE_FAS_TEST','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2))
print('FAS_DEVELOPMENT_FROZEN',flush=True)
PY
