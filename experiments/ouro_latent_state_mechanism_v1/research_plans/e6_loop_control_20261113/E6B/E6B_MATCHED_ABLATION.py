"""Predeclared matched output-only/latent contrasts, no fitting or selection."""
import argparse,json
from pathlib import Path
from E6B_ANALYSIS import controller_metrics,write,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.results);freeze=json.loads((root/'E6B_FROZEN_SPEC.json').read_text());specs=freeze['zero_label_specs']+freeze['selected_cal_specs'];rows=[]
 for name in ['E6B_DEVELOPMENT_DECISIONS.jsonl','E6B_CONTROLLER_RESULTS.jsonl']:
  f=root/name
  if f.exists():rows.extend(map(json.loads,f.read_text().splitlines()))
 result=[]
 for split in ['train','validation','test']:
  data=[r for r in rows if r['split']==split]
  if not data:continue
  for lo,hi in [(0,1),(2,3),(4,5),(6,7),(8,9)]:
   out={r['id']:r for r in data if r['method_index']==lo}
   latent={r['id']:r for r in data if r['method_index']==hi};assert set(out)==set(latent);ids=sorted(out)
   m=controller_metrics([{'gold':latent[k]['gold']} for k in ids],[latent[k] for k in ids],[out[k]['prediction'] for k in ids]);m.update({'split':split,'output_only_spec':specs[lo],'latent_spec':specs[hi],'scope':'Same fixed strategy ZL pairs, or separately Validation-selected Cal pair; no Test selection.'});result.append(m)
 write(a.out,{'source_sha256':sha(__file__),'frozen_spec_sha256':sha(root/'E6B_FROZEN_SPEC.json'),'comparisons':result});print('MATCHED_ABLATION_COMPLETE',len(result))
if __name__=='__main__':main()
