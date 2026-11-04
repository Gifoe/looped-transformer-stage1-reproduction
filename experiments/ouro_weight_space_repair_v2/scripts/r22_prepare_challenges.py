"""Freeze challenge IDs using native/R6 winners only, before any feature."""
import json,numpy as np
from r22_common import ROOT,read,save,digest,sha

def main():
 audit=read('results/slcd_SOURCE_AUDIT.json');states={s['id']:s for s in audit['states']};result=[]
 for split,n in [('discovery',384),('selection',192)]:
  a=[json.loads(x) for x in (ROOT/('results/slcd_r2_'+split+'_development.jsonl')).read_text(encoding='utf8').splitlines()];b=[json.loads(x) for x in (ROOT/('results/r7_'+split+'_development.jsonl')).read_text(encoding='utf8').splitlines()];assert len(a)==len(b)==n
  r7={r['id']:r for r in b};items={r['id']:r for r in [json.loads(x) for x in (ROOT/('data/slcd_'+split+'.jsonl')).read_text(encoding='utf8').splitlines()]}
  for x in a:
   y=r7[x['id']];state=states[x['id']];assert state['split']==split and x['choice_token_counts']==state['choice_token_counts']==y['token_counts']
   native=[int(np.argmax(x['original_methods']['native_k'+str(k)]['normalized_scores'])) for k in range(1,7)];r6=int(np.argmax(y['baseline_scores']['M0_R6']));challengers=sorted(set(native)-{r6})
   result.append(dict(id=x['id'],split=split,R6_winner=r6,native_winners=native,challengers=challengers,request_pairs=items[x['id']]['request_pairs'],native6_scores=x['original_methods']['native_k6']['normalized_scores'],state=state))
 assert len(result)==576
 save('configs/r22_CHALLENGE_MANIFEST.json',dict(status='label_free_pre_feature',n=576,rows=result,rows_sha256=digest(result),input_sha256={p:sha(p) for p in ['results/slcd_SOURCE_AUDIT.json']+['results/'+prefix+split+'_development.jsonl' for prefix in ['slcd_r2_','r7_'] for split in ['discovery','selection']]},scope='No gold fields used; unique actual native winners excluding R6; no trajectory feature computed'))
 print('CHALLENGE_FROZEN',576)
if __name__=='__main__':main()

