"""Isolated exposed-Val fixture, only after H2 Train freeze; never reads Test labels/files."""
from pathlib import Path
import json,hashlib,datetime
root=Path('/root/autodl-tmp/ouro-e7');e6=Path('/root/autodl-tmp/ouro-e6a');out=root/'analysis';assert (out/'H2_TRAIN_FROZEN_SPEC.json').exists();target=root/'H2_VALIDATION_ONLY_INPUTS.jsonl';assert not target.exists()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=e6/'E6B/joint5090/E6B_DEVELOPMENT_DECISIONS.jsonl';rows=[json.loads(l) for l in source.read_text().splitlines()];assert all(r['split'] in ['train','validation'] for r in rows);gold={r['id']:r['gold'] for r in rows if r['split']=='validation' and r['method_index']==0};assert len(gold)==299;outrows=[]
for key in sorted(gold):
 f=e6/'joint/validation'/('validation_'+hashlib.sha256(key.encode()).hexdigest()+'.json');n=json.loads(f.read_text());assert n['split']=='validation';outrows.append({'id':key,'split':'validation','is_demonstration':False,'gold':gold[key],'group':n['input_ids_sha256'],'outer_fold':None,'inner_fold':None,'input_ids_sha256':n['input_ids_sha256'],'capture_sha256':sha(f),'scores':[s['scores'] for s in n['native']],'pred':[s['prediction'] for s in n['native']]})
target.write_text(''.join(json.dumps(r)+'\n' for r in outrows));(root/'H2_VALIDATION_EXPORT_RECEIPT.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'COMPLETE','n':299,'source_sha256':sha(source),'fixture_sha256':sha(target),'exporter_source_sha256':sha(Path(__file__)),'train_frozen_spec_sha256':sha(out/'H2_TRAIN_FROZEN_SPEC.json'),'Test_inputs_loaded':False},indent=2));print('H2_VALIDATION_FIXTURE_EXPORTED',sha(target))
