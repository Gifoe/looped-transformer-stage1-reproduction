"""Independent E7 raw-unit grid/correctness/contrast audit, CPU only."""
from pathlib import Path
from fractions import Fraction as F
import argparse,collections,hashlib,json
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()
def labels(p):
 return {r['id']:r['gold'] for r in (json.loads(line) for line in Path(p).read_text().splitlines())}
def exact_contrasts(matrix):
 m=len(matrix);allp0=sum(matrix[0])/4;allnonp=sum(sum(x) for x in matrix[1:])/(4*(m-1))
 s0=sum(x[0] for x in matrix)/m;non_s=sum(sum(x[1:]) for x in matrix)/(3*m)
 i=sum(sum(x[1:]) for x in matrix[1:])/(3*(m-1))-sum(x[0] for x in matrix[1:])/(m-1)-sum(matrix[0][1:])/3+matrix[0][0]
 return {'P':allnonp-allp0,'S':non_s-s0,'interaction':i}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
 assert read(r/'full_capture/COMPLETE.json')['status']=='COMPLETE'
 labelpaths={'train':'/root/autodl-tmp/ouro-e6a/next_step_train_diagnosis/TRAIN_ONLY_INPUTS.jsonl',
             'validation':'/root/autodl-tmp/ouro-e7/H2_VALIDATION_ONLY_INPUTS.jsonl'}
 result={'status':'PASS','source_sha256':sha(__file__),'full_capture_receipt_sha256':sha(r/'full_capture/COMPLETE.json'),
         'scope':'Raw unit checksum, mapping, first ties, per-depth predictions, gate depth/readout, complete grid, exact question-block primary means and fixed pair correctness; not a second GPU or bootstrap run.','splits':{}}
 for split in ['train','validation']:
  gold=labels(labelpaths[split]);groups=collections.defaultdict(dict);n=0
  for path in sorted((r/'full_capture').glob(split+'_*.json')):
   unit=read(path);checksum=unit.pop('unit_payload_sha256');assert canonical(unit)==checksum
   assert unit['status']=='COMPLETE' and unit['split']==split and unit['id'] in gold
   m=len(unit['targets']);p0=unit['p'];s0=unit['s'];mapping=unit['display_to_original']
   assert mapping==[(i+p0)%m for i in range(m)]
   assert unit['weight_versions_unchanged'] and unit['production_physical_loop_applications']==4
   assert unit['production_actual_counts']=={'model_body_forwards':1,'head_calls':4,'first_layer_calls':4}
   assert len(unit['native'])==4
   for depth,row in enumerate(unit['native'],1):
    candidate=row['candidate_logits'];tie=[i for i,v in enumerate(candidate) if v==max(candidate)]
    assert row['depth']==depth and row['prediction_display']==tie[0]
    assert row['prediction_semantic']==mapping[tie[0]]
    assert row['top_display_indices']==tie and row['top_semantic_indices']==sorted(mapping[i] for i in tie)
   for threshold in [.25,.5,.75,1.]:
    depth=next((i+1 for i,x in enumerate(unit['gate_R']['exit_cdf']) if x>=threshold),4)
    selected=unit['gate_R']['threshold_readouts'][str(threshold)]
    assert selected['selected_depth']==depth and selected['full_logits_sha256']==unit['native'][depth-1]['full_logits_sha256']
   assert (p0,s0) not in groups[unit['id']];groups[unit['id']][p0,s0]=unit;n+=1
  assert set(groups)==set(gold) and n==({'train':17504,'validation':4776}[split])
  effects=collections.defaultdict(lambda:collections.defaultdict(list));paircounts=collections.Counter();nativecounts=collections.Counter()
  for ident,views in groups.items():
   m=len(views[0,0]['targets']);assert set(views)=={(p0,s0) for p0 in range(m) for s0 in range(4)}
   g=[[F(int(views[p0,s0]['native'][1]['prediction_semantic']==gold[ident])-int(views[p0,s0]['native'][0]['prediction_semantic']==gold[ident])) for s0 in range(4)] for p0 in range(m)]
   for effect,value in exact_contrasts(g).items():effects[m][effect].append(value)
   lp=[]
   for p0 in [0,m//2]:
    u=views[p0,0];v=np.asarray(u['native'][1]['scores'],dtype=np.float64);semantic=np.empty(m)
    for display,original in enumerate(u['display_to_original']):semantic[original]=v[display]
    lp.append(semantic-semantic.mean())
   pairpred=int(np.argmax((lp[0]+lp[1])/2));paircounts[m]+=pairpred==gold[ident]
   nativecounts[m]+=views[0,0]['native'][3]['prediction_semantic']==gold[ident]
  reported=read(r/'analysis'/(split.upper()+'_CAUSAL_CONTRASTS.json'));pair=read(r/'analysis'/(split.upper()+'_FIXED_PAIR_SUMMARY.json'))
  report={}
  for m,terms in effects.items():
   report[str(m)]={'n':len(terms['P']),'exact_means':{},'fixed_pair_correct':int(paircounts[m]),'native_K4_correct':int(nativecounts[m])}
   for effect,values in terms.items():
    mean=sum(values)/len(values);target=next(x for x in reported if x['primary'] and x['stratum']==str(m)+'_choice' and x['tie_rule']=='pred' and x['effect']==effect)
    assert abs(float(mean)-target['estimate'])<1e-14
    report[str(m)]['exact_means'][effect]=str(mean)
   target=next(x for x in pair if x['m']==m and x['reference']=='identity_K4')
   assert target['correct']==paircounts[m] and target['reference_correct']==nativecounts[m]
  result['splits'][split]={'questions':len(groups),'views':n,'isolated_labels_sha256':sha(labelpaths[split]),'strata':report}
 out=r/'analysis/ROOT_INDEPENDENT_ARITHMETIC_AUDIT.json';out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
