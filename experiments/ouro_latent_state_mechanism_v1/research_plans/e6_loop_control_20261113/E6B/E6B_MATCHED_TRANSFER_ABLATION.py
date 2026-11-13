"""No-selection matched latent/output-only contrasts in frozen FAS transfer."""
import argparse,json
from pathlib import Path
from E6B_ANALYSIS import controller_metrics,write,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--joint-freeze',required=True);p.add_argument('--decisions',required=True);p.add_argument('--out',required=True);a=p.parse_args();freeze=json.loads(Path(a.joint_freeze).read_text());specs=freeze['zero_label_specs']+freeze['selected_cal_specs'];rows=list(map(json.loads,Path(a.decisions).read_text().splitlines()));result=[]
 for split in ['train','validation','test']:
  for lo,hi in [(0,1),(2,3),(4,5),(6,7),(8,9)]:
   plain={r['id']:r for r in rows if r['split']==split and r['method_index']==lo};latent={r['id']:r for r in rows if r['split']==split and r['method_index']==hi};assert set(plain)==set(latent);ids=sorted(plain);assert ids
   m=controller_metrics([{'gold':latent[k]['gold']} for k in ids],[latent[k] for k in ids],[plain[k]['prediction'] for k in ids]);m.update(split=split,output_only_spec=specs[lo],latent_spec=specs[hi],score_protocol='full_answer_character_mean');result.append(m)
 write(a.out,{'source_sha256':sha(__file__),'joint_freeze_sha256':sha(a.joint_freeze),'scope':'Joint-frozen identical rules and coefficients; no FAS fitting or selection','comparisons':result});print('MATCHED_FAS_TRANSFER_COMPLETE',len(result))
if __name__=='__main__':main()
