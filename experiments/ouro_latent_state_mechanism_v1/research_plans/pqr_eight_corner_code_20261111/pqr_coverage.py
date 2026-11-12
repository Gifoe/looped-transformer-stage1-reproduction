"""Independent CUDA replay and old-corner validation; draft awaiting review."""
import json,hashlib,argparse
from pathlib import Path
if not __debug__:raise RuntimeError('optimized Python forbidden')
import torch,pyarrow.parquet as pq
from safetensors.torch import load_file
from pqr_runner import ROOT,ORIG,RESUME,OLD,BOUNDARY,bindings,sha,split_data_path,canonical,write_once,validate_envelope,partition,runtime_quick,watched_paths
from pqr_core import NEW,OLD as OLD_NAMES,CORNERS
from pqr_integrity import validate_attempt,binding_guard

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--split',required=True,choices=('train','validation','test'));ap.add_argument('--shards',required=True);ap.add_argument('--binding',required=True);ap.add_argument('--report',required=True);ap.add_argument('--probe-id');a=ap.parse_args();d=Path(a.shards).resolve();d.relative_to(ROOT.resolve());out=Path(a.report).resolve();out.relative_to(ROOT.resolve());assert not out.exists()
 binding_path=Path(a.binding).resolve();binding_sha=sha(binding_path);frozen=json.loads(binding_path.read_text());assert sha(binding_path)==binding_sha;assert bindings()==frozen['binding'];receipt=json.loads((d/'_receipt.json').read_text());assert receipt['binding_sha256']==binding_sha and receipt['status']==('PQR_PROBE_PENDING_REPLAY' if a.probe_id else 'PQR_SPLIT_PENDING_COVERAGE')
 split=a.split;nv=json.loads(((RESUME if split=='test' else ORIG)/('coverage_'+split+'.json')).read_text());ov=json.loads((OLD/('coverage_'+split+'.json')).read_text());natural={q['id']:q for q in nv['questions']};old={q['id']:q for q in ov['questions']};ledger={}
 for line in BOUNDARY.open():
  row=json.loads(line)
  if row['split']==split:assert row['id'] not in ledger;ledger[row['id']]=row
 rows=pq.read_table(split_data_path(split)).to_pylist();assert len(rows)==len(old)==len(natural)==len(ledger) and {r['id'] for r in rows}==set(old)==set(natural)==set(ledger)
 if a.probe_id:assert frozen['longest_request']['id']==a.probe_id and frozen['longest_request']['split']==split;rows=[r for r in rows if r['id']==a.probe_id];assert len(rows)==1
 assert receipt['split']==split and receipt['questions']==len(rows);keys={hashlib.sha256(r['id'].encode()).hexdigest() for r in rows};assert {p.name for p in d.iterdir()}=={'_receipt.json'}|{k+'.json' for k in keys}|{k+'.safetensors' for k in keys};assert torch.cuda.is_available() and torch.cuda.is_bf16_supported();entries=[];replays=0
 runtime_snapshot=runtime_quick();assert runtime_snapshot==frozen['binding']['runtime'];watched=watched_paths(frozen['binding'])
 def quick_guard():
  assert sha(binding_path)==binding_sha and runtime_quick()==runtime_snapshot and all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in watched.items())
 for row in rows:
  quick_guard();ident=row['id'];key=hashlib.sha256(ident.encode()).hexdigest();nf=ORIG/'shards'/split/(key+'.json');assert sha(nf)==natural[ident]['metadata_sha256'];raw=nf.read_bytes();meta=json.loads(raw);regions=partition(raw,natural[ident]['metadata_sha256'],ledger[ident]);records=sorted(meta['records'],key=lambda r:r['candidate']);targets=[r['encoding']['target_ids'][0] for r in records];oldf=OLD/split/(key+'.json');oldt=oldf.with_suffix('.safetensors');assert sha(oldf)==old[ident]['metadata_sha256'] and sha(oldt)==old[ident]['logits_sha256'];oldj=json.loads(oldf.read_text());idn=oldj['payload']['identity'];validate_envelope(oldf,idn,meta['gold_index'],targets,records);assert {k:idn[k] for k in ov['source']}==ov['source']
  f=d/(key+'.json');j=json.loads(f.read_text());p=j['payload'];assert j['payload_sha256']==hashlib.sha256(canonical(p)).hexdigest() and p['id']==ident and p['split']==split and p['binding_sha256']==binding_sha and p['old_metadata_sha256']==sha(oldf) and p['old_logits_sha256']==sha(oldt) and p['natural_metadata_sha256']==sha(nf);validate_attempt(p,binding_sha,ROOT,frozen['longest_request'],bool(a.probe_id),sha);tf=f.with_suffix('.safetensors');assert p['logits_sha256']==sha(tf);v=p['result'];assert v['regions']==regions and v['gold']==meta['gold_index'] and v['target_ids']==targets and set(v['corners'])==set(CORNERS) and v['anchors']=={'original_four':'exact','native_K1_K2':'exact','last_layer_C':'exact'}
  base=oldj['payload']['result'];assert v['fixed_wrong']==base['k1_highest_wrong'];z=load_file(str(tf));assert set(z)=={'corner_'+x for x in NEW}
  for corner,oldname in OLD_NAMES.items():assert v['corners'][corner]==base['conditions'][oldname]
  for corner in NEW:
   t=z['corner_'+corner];assert t.dtype==torch.bfloat16 and tuple(t.shape)==(1,49152) and torch.isfinite(t).all();t=t.to('cuda');scores=[float(t.float().log_softmax(-1).gather(-1,torch.tensor([[target]],device='cuda')).sum()) for target in targets];r=v['corners'][corner];assert r['scores']==scores;pred=max(range(len(scores)),key=lambda i:(scores[i],-i));assert r['prediction']==pred and r['top_tie']==(sum(x==scores[pred] for x in scores)>1) and r['gold_vs_fixed_wrong_gap']==scores[v['gold']]-scores[v['fixed_wrong']];replays+=1
  quick_guard();entries.append({'id':ident,'metadata_sha256':sha(f),'logits_sha256':sha(tf)})
 assert sha(binding_path)==binding_sha and bindings()==frozen['binding'];report={'status':'PQR_LONGEST_PROBE_CUDA_REPLAY_ZERO' if a.probe_id else 'PQR_COMPLETE_CUDA_REPLAY_ZERO','split':split,'expected_questions':len(rows),'complete_questions':len(entries),'replayed_condition_logits':replays,'binding_sha256':binding_sha,'receipt_sha256':sha(d/'_receipt.json'),'questions':entries};write_once(out,report);print('PQR_COVERAGE_COMPLETE',split,len(entries),replays)
if __name__=='__main__':main()
