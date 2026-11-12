"""Read-only, per-question ledger; missing or corrupt shards are never counted complete."""
import argparse,hashlib,json,math,os
from pathlib import Path
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file
from cohort_runner import write_new_json
from cohort_runner import PROJECT,split_data_path,sha,write_json,bound_hashes
from manifest_guard import validate_manifest

def main():
 if not __debug__:raise RuntimeError('coverage audit refuses Python optimization; assertions required')
 ap=argparse.ArgumentParser();ap.add_argument('--split',required=True,choices=['train','validation','test']);ap.add_argument('--shards',required=True);ap.add_argument('--out',required=True);ap.add_argument('--score-device',choices=['cuda','cpu'],default='cuda');a=ap.parse_args()
 if a.score_device=='cpu' and os.environ.get('OURO_SYNTHETIC_TEST')!='1':raise RuntimeError('CPU replay is restricted to explicit synthetic tests')
 if a.score_device=='cuda':
  if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 score replay required')
  free,total=torch.cuda.mem_get_info()
  if free<40*1024**3 or free/total<0.90:raise RuntimeError('GPU not sufficiently free for non-interfering score replay')
 root=Path(a.shards).resolve();root.relative_to(PROJECT.resolve());out=Path(a.out).resolve();out.relative_to(PROJECT.resolve())
 rows=pq.read_table(split_data_path(a.split)).to_pylist();ids=[x['id'] for x in rows]
 if len(set(ids))!=len(ids):raise RuntimeError('duplicate input IDs')
 manifest=Path(__file__).parent/'results'/f'INPUT_MANIFEST_{a.split}.jsonl';requests={}
 validate_manifest(a.split,split_data_path(a.split),Path(__file__).parent/'results/INPUT_MANIFEST_SUMMARY.json',manifest)
 for line in manifest.read_text(encoding='utf8').splitlines():
  r=json.loads(line);key=(r['id'],r['candidate'])
  if key in requests:raise RuntimeError('duplicate manifest request')
  requests[key]=r
 current_hashes=bound_hashes(a.split)
 ledger=[];counts={'expected_questions':len(rows),'expected_requests':sum(len(x['choices']['text']) for x in rows),'expected_loop_states':4*len(requests),'expected_layer_states':96*len(requests),'expected_selected_token_layer_vectors':sum(r['selected_token_count']*96 for r in requests.values()),'complete_questions':0,'complete_requests':0,'complete_loop_states':0,'complete_layer_states':0,'complete_selected_token_layer_vectors':0,'missing_questions':0,'failed_questions':0}
 for row in rows:
  ident=row['id'];name=hashlib.sha256(ident.encode()).hexdigest();meta_path=root/(name+'.json');tensor_path=root/(name+'.safetensors');entry={'id':ident,'split':a.split,'expected_requests':len(row['choices']['text']),'status':'failed','errors':[]}
  if not meta_path.exists() or not tensor_path.exists():
   entry.update(status='missing',errors=['missing committed metadata or tensor']);counts['missing_questions']+=1;ledger.append(entry);continue
  try:
   m=json.loads(meta_path.read_text(encoding='utf8'));assert m['id']==ident and m['split']==a.split;assert m['sha256']==current_hashes,'stale source/model/environment hash';assert sha(tensor_path)==m['tensor_sha256'];assert len(m['records'])==entry['expected_requests']
   if a.score_device=='cpu' and m.get('runtime',{}).get('gpu'):raise RuntimeError('CPU replay cannot audit GPU-origin production shards')
   if a.score_device=='cuda':
    runtime=m['runtime']
    if runtime['gpu']!=torch.cuda.get_device_name() or runtime['torch']!=torch.__version__ or runtime['cuda']!=torch.version.cuda or runtime['dtype']!='bfloat16' or runtime['attention']!='sdpa' or runtime['use_cache'] is not False:raise RuntimeError('score replay hardware or numerical protocol differs from collection')
   ts=load_file(str(tensor_path));expected_keys=set()
   for j,record in enumerate(m['records']):
    ref=requests[(ident,j)];assert record['candidate']==j and record['input_ids_sha256']==ref['input_sha256'];enc=record['encoding'];pos=record['captured_positions'];assert pos==sorted(set(sum(enc['spans'].values(),[])+enc['predecessor_positions'])) and len(pos)==ref['selected_token_count'];assert len(record['scores'])==4 and all(math.isfinite(x) for x in record['scores']);assert record['event_sequence']==[[k,i] for k in range(4) for i in range(24)]
    for k in range(1,5):
     for layer in range(24):
      key=f'candidate{j}_loop{k}_layer{layer:02d}';expected_keys.add(key);x=ts[key];assert x.dtype==torch.bfloat16 and tuple(x.shape)==(len(pos),2048) and bool(torch.isfinite(x).all())
     key=f'candidate{j}_loop{k}_post_norm';expected_keys.add(key);x=ts[key];assert x.dtype==torch.bfloat16 and tuple(x.shape)==(len(pos),2048) and bool(torch.isfinite(x).all())
     key=f'candidate{j}_loop{k}_lm_head_logits';expected_keys.add(key);x=ts[key];assert x.dtype==torch.bfloat16 and tuple(x.shape)==(len(enc['target_ids']),49152) and bool(torch.isfinite(x).all())
     target=torch.tensor(enc['target_ids'],dtype=torch.long,device=a.score_device);replayed=float(x.to(a.score_device).float().log_softmax(-1).gather(-1,target[:,None]).sum())
     if replayed!=record['scores'][k-1]:raise RuntimeError(f'score replay mismatch candidate={j} loop={k} stored={record["scores"][k-1]} replayed={replayed} abs_error={abs(replayed-record["scores"][k-1])}')
   assert set(ts)==expected_keys
   gold=row['choices']['label'].index(row['answerKey']);assert m['gold_index']==gold
   ranks=[sorted(range(len(m['records'])),key=lambda j:-m['records'][j]['scores'][k]) for k in range(4)]
   predictions=[order[0] for order in ranks]
   margins=[m['records'][gold]['scores'][k]-max(m['records'][j]['scores'][k] for j in range(len(m['records'])) if j!=gold) for k in range(4)]
   assert m['rankings']==ranks and m['predictions']==predictions and m['gold_margins']==margins
   entry.update(status='complete',actual_requests=len(m['records']),tensor_sha256=m['tensor_sha256'],metadata_sha256=sha(meta_path));counts['complete_questions']+=1;counts['complete_requests']+=len(m['records']);counts['complete_loop_states']+=4*len(m['records']);counts['complete_layer_states']+=96*len(m['records']);counts['complete_selected_token_layer_vectors']+=96*sum(len(r['captured_positions']) for r in m['records'])
  except Exception as e:entry['errors'].append(type(e).__name__+': '+str(e));counts['failed_questions']+=1
  ledger.append(entry)
 write_new_json(out,{'ledger_schema':'ouro-coverage-v2','audit_source_sha256':sha(Path(__file__)),'ledger_guard_sha256':sha(Path(__file__).with_name('ledger_guard.py')),'scope':'shard coverage and score replay; not scientific interpretation','score_replay':{'device':a.score_device,'status':'passed' if counts['complete_questions']==counts['expected_questions'] else 'incomplete','tolerance':0.0,'torch':torch.__version__,'cuda':torch.version.cuda if a.score_device=='cuda' else None,'gpu':torch.cuda.get_device_name() if a.score_device=='cuda' else None},'counts':counts,'questions':ledger})
 if counts['complete_questions']!=counts['expected_questions']:raise SystemExit('coverage incomplete')
if __name__=='__main__':main()
