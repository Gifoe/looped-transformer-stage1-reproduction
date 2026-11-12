"""Train-only BF16 target logits export; no validation/test data."""
import json,hashlib,gzip
from pathlib import Path
from safetensors.torch import load_file
ROOT=Path('/home/luansiyu/loop-latent');RUN=ROOT/'orchestration_runs/pqr-round3-20261111';OLD=ROOT/'orchestration_runs/round2-full-20261111';OUT=ROOT/'research_plans/round3_train_exploratory_20261111'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
cv=RUN/'coverage_train.json';assert sha(cv)=='93654dd3a4a53312029e2b33f37253043b5c3086a9330e70571fc47c259dd72d';c=json.loads(cv.read_text());oc=json.loads((OLD/'coverage_train.json').read_text());old={q['id']:q for q in oc['questions']};assert len(old)==len(c['questions'])==1119
oldmap={'111':'core_K1C_K1R','112':'core_K1C_K2R','221':'core_K2C_K1R','222':'core_K2C_K2R'};names=('111','112','121','122','211','212','221','222')
out=OUT/'candidate_logits.jsonl.gz'
with gzip.open(out,'xt',encoding='utf8') as f:
 for q in c['questions']:
  ident=q['id'];key=hashlib.sha256(ident.encode()).hexdigest();np=RUN/'train'/(key+'.json');nt=np.with_suffix('.safetensors');op=OLD/'train'/(key+'.json');ot=op.with_suffix('.safetensors')
  assert sha(np)==q['metadata_sha256'] and sha(nt)==q['logits_sha256'] and sha(op)==old[ident]['metadata_sha256'] and sha(ot)==old[ident]['logits_sha256']
  j=json.loads(np.read_text())['payload']['result'];o=json.loads(op.read_text())['payload']['result'];targets=j['target_ids'];a=load_file(str(nt));b=load_file(str(ot));vals={}
  for name in names:
   z=b[oldmap[name]] if name in oldmap else a['corner_'+name]
   assert str(z.dtype)=='torch.bfloat16' and list(z.shape)==[1,49152]
   vals[name]=[float(z[0,t]) for t in targets]
  record={'id':ident,'target_ids':targets,'gold':j['gold'],'fixed_wrong':j['fixed_wrong'],'target_logits':vals,'scores':{name:j['corners'][name]['scores'] for name in names},'pqr_metadata_sha256':sha(np),'pqr_logits_sha256':sha(nt),'round2_metadata_sha256':sha(op),'round2_logits_sha256':sha(ot)}
  f.write(json.dumps(record,sort_keys=True)+'\n')
receipt={'status':'TRAIN_TARGET_BF16_LOGITS_EXPORT_ONLY','questions':1119,'pqr_coverage_sha256':sha(cv),'round2_coverage_sha256':sha(OLD/'coverage_train.json'),'output_sha256':sha(out),'script_sha256':sha(Path(__file__))}
with (OUT/'candidate_logits_receipt.json').open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
print('TRAIN_LOGITS_READY',receipt['output_sha256'],sha(OUT/'candidate_logits_receipt.json'),flush=True)
