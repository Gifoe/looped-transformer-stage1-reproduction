"""One deterministic 64-nonce pool; CPU tokenizer structure only, no model."""
from pathlib import Path
import json,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PROPOSAL_SHA='5f2b13c148c58ce38a34196a6cccd04e0204ea2056b64bd98401fd36a1791df7'
TEMPLATE='Use the lookup table below. Each code maps to exactly one label.\n{rows}\nFor code {query}, what is its label? Reply with only A, B, C, or D.'
SUFFIX='For code {nonce}, what is its label? Reply with only A, B, C, or D.\nAnswer:'
def sha(p):return hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def save(p,x):
 with (ROOT/p).open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def generate(tok):
 rng=np.random.default_rng(2021);alphabet=list('abcdefghijklmnopqrstuvwxyz0123456789');pools=[['zx'+''.join(rng.choice(alphabet,size=6).tolist()) for _ in range(32)] for _ in range(2)]
 output=dict(pools=pools,seed=2021,selection='Lexicographically first (i,j) indices with equal literal suffix token length, then earliest two unused; one finite draw only',tables=[],errors=[])
 if len(set(sum(pools,[])))!=64:output['errors'].append('Duplicate among64 finite pool entries; no redraw');output['status']='no_admission_no_template_or_pool_retry';return output
 prefix=json.loads((ROOT/'configs/r25_MEMORY_MANIFEST.json').read_text())['key_text'];assert prefix.endswith('Question:');prefix=prefix[:-len('Question:')]
 output['fixed25_prefix_utf8_sha256']=hashlib.sha256(prefix.encode()).hexdigest()
 from runtime import encode_pair
 for table,pool in enumerate(pools):
  lengths=[len(tok.encode(SUFFIX.format(nonce=n),add_special_tokens=False)) for n in pool];pair=next(((i,j) for i in range(32) for j in range(i+1,32) if lengths[i]==lengths[j]),None)
  record=dict(table=table+1,pool_suffix_lengths=lengths,query_pair_indices=pair,instances=[]);output['tables'].append(record)
  if pair is None:output['errors'].append(f'Table{table+1}: no equal-length pair in32');continue
  others=[i for i in range(32) if i not in pair][:2];indices=list(pair)+others if table==0 else others+list(pair);keys=[pool[i] for i in indices];query_rows=[0,1] if table==0 else [2,3];record.update(row_pool_indices=indices,row_keys=keys,query_rows=query_rows)
  for qr in query_rows:
   variants=[]
   for version in ['A','B']:
    labels=list('ABCD')
    if version=='B':a,b=query_rows;labels[a],labels[b]=labels[b],labels[a]
    lines=[k+' -> '+label for k,label in zip(keys,labels)];question=TEMPLATE.format(rows='\n'.join(lines),query=keys[qr]);prompt=prefix+'Question: '+question+'\nAnswer:'
    assert not any(line.rstrip()!=line for line in question.splitlines())
    encoded=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);ids=encoded['input_ids'];offsets=encoded['offset_mapping'];sources=[]
    cursor=len(prefix+'Question: '+'Use the lookup table below. Each code maps to exactly one label.\n')
    for row,(key,label,line) in enumerate(zip(keys,labels,lines)):
     start=cursor+len(key)+3;end=start+2;assert prompt[start:end]==' '+label
     overlap=[i for i,(a,b) in enumerate(offsets) if b>a and a<end and b>start];contained=[i for i in overlap if start<=offsets[i][0] and offsets[i][1]<=end]
     if len(overlap)!=1 or overlap!=contained:output['errors'].append(f'Table{table+1} query{qr} {version} row{row}: label source not one fully-contained token')
     position=contained[0] if len(contained)==1 else None
     p,t=encode_pair(tok,prompt,' '+label)
     if p!=ids or len(t)!=1:output['errors'].append(f'Table{table+1} query{qr} {version} label{label}: original encode_pair boundary not one token')
     if position is not None and len(t)==1 and ids[position]!=t[0]:output['errors'].append(f'Table{table+1} query{qr} {version} label{label}: source/continuation token mismatch')
     sources.append(dict(row=row,label=label,field_start=start,field_end=end,field_text=prompt[start:end],position=position,offset=list(offsets[position]) if position is not None else None,token_id=ids[position] if position is not None else None,continuation_ids=t,continuation_prefix_ids_sha256=digest(p)))
     cursor+=len(line)+1
    query_start=len(prefix+'Question: ')+len(question)-len(('For code '+keys[qr]+', what is its label? Reply with only A, B, C, or D.'))
    if any(x['field_start']<len(prefix) or x['field_end']>query_start for x in sources):output['errors'].append('Source outside current table or overlaps query')
    positions=[x['position'] for x in sources]
    if None in positions or len(set(positions))!=4:output['errors'].append('Four distinct source token positions unavailable')
    variants.append(dict(version=version,prompt=prompt,prompt_utf8_sha256=hashlib.sha256(prompt.encode()).hexdigest(),input_ids=ids,input_ids_sha256=digest(ids),input_shape=[1,len(ids)],source_positions=positions,sources=sources,query_start=query_start,expected_label=labels[qr],expected_label_metadata_only=True,no_expected_answer_appended=True))
   a,b=variants;changed=[i for i,(x,y) in enumerate(zip(a['input_ids'],b['input_ids'])) if x!=y]
   expected_positions=[a['source_positions'][i] for i in query_rows]
   if a['input_shape']!=b['input_shape'] or a['source_positions']!=b['source_positions'] or set(changed)!=set(expected_positions):output['errors'].append(f'Table{table+1} query{qr}: A/B shape/source/difference contract failed')
   record['instances'].append(dict(query_row=qr,query_key=keys[qr],variants=variants,changed_positions=changed,expected_changed_positions=expected_positions))
  instances=record['instances']
  for version in [0,1]:
   a,b=[q['variants'][version] for q in instances]
   if a['input_shape']!=b['input_shape'] or a['source_positions']!=b['source_positions']:output['errors'].append(f'Table{table+1}: cross-query shape/source contract failed')
 output['status']='passed_CPU_structural_admission_only' if not output['errors'] else 'no_admission_no_template_or_pool_retry'
 return output

def main():
 started=time.monotonic();assert sha('r27_FUNCTIONAL_ADMISSION_PROPOSAL.md')==PROPOSAL_SHA
 from transformers import AutoTokenizer
 old=json.loads((ROOT/'configs/r25_HELLINGER_MEMORY_LOCK.json').read_text());meta=old['model_identity']['metadata_and_tokenizer_sha256']
 assert all(sha('models/Ouro-1.4B/'+p)==h for p,h in meta.items())
 tok=AutoTokenizer.from_pretrained(str(ROOT/'models/Ouro-1.4B'),local_files_only=True,use_fast=True);assert tok.is_fast
 out=generate(tok);out.update(proposal_sha256=PROPOSAL_SHA,authorization_sha256=sha('r27_CPU_STRUCTURE_AUTHORIZATION.md'),source_sha256=sha('scripts/r27_cpu_structure.py'),tokenizer_metadata_sha256=meta,original_encode_pair_source_sha256=sha('scripts/runtime.py'),fixed25_memory_manifest_sha256=sha('configs/r25_MEMORY_MANIFEST.json'))
 save('configs/r27_CPU_STRUCTURE_MANIFEST.json',out)
 save('results/r27_CPU_STRUCTURE_RECEIPT.json',dict(status=out['status'],errors=out['errors'],manifest_sha256=sha('configs/r27_CPU_STRUCTURE_MANIFEST.json'),source_sha256=sha('scripts/r27_cpu_structure.py'),elapsed_CPU_wall_seconds=time.monotonic()-started,model_loaded=False,GPU_worker_seconds=0,ARC_gold_predictions_scores_hidden_or_task_accuracy_accessed=False,finite_pools=2,candidates_per_pool=32,retries=0,scope='Tokenizer-only structural selection, not functional evidence or GPU authorization'))
 print(out['status']);print(json.dumps(out['errors']))
if __name__=='__main__':main()
