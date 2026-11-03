"""Question-paired summaries and export figures; runs on CPU after scoring."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from runtime import ROOT,save,configure
import os
configure();os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')

def writecsv(path,rows,fields=None):
 path.parent.mkdir(parents=True,exist_ok=True)
 fields=fields or list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def paired(records,before,after,split):
 from scipy.stats import binomtest
 b=np.array([r['methods'][before]['correct'] for r in records],dtype=int)
 a=np.array([r['methods'][after]['correct'] for r in records],dtype=int)
 dm=np.array([r['methods'][after]['margin']-r['methods'][before]['margin'] for r in records])
 d=a-b;rescue=int(np.sum(d==1));damage=int(np.sum(d==-1));n=len(d)
 assert n>0 and np.isfinite(dm).all()
 rng=np.random.default_rng(2021);accs=[];margins=[]
 for _ in range(100):
  idx=rng.integers(n,size=(100,n));accs.extend(d[idx].mean(1));margins.extend(dm[idx].mean(1))
 return dict(split=split,before=before,after=after,n=n,rescues=rescue,regressions=damage,discordant_pairs=rescue+damage,net_correct=rescue-damage,delta_accuracy_pp=100*float(d.mean()),bootstrap_accuracy_pp_low=100*float(np.quantile(accs,.025)),bootstrap_accuracy_pp_high=100*float(np.quantile(accs,.975)),mean_delta_margin=float(dm.mean()),bootstrap_margin_low=float(np.quantile(margins,.025)),bootstrap_margin_high=float(np.quantile(margins,.975)),exact_mcnemar_p=1. if rescue+damage==0 else float(binomtest(rescue,rescue+damage,.5).pvalue),bootstrap_replicates=10000,bootstrap_seed=2021,bootstrap_unit='question',bootstrap_interval='pointwise_95_percentile_not_multiplicity_adjusted',inference_status='exploratory_secondary_no_primary_weight_edit')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--include-test',action='store_true');a=ap.parse_args()
 splits=['discovery','selection','validation']+(['test'] if a.include_test else [])
 if a.include_test:
  final=json.loads((ROOT/'configs/FINAL_LOCK.json').read_text(encoding='utf8'))
  assert final['test_evaluation_authorized'] and final['primary_weight_edit'] is None
 allrows={};summary=[];pairs=[];contrasts=[]
 for split in splits:
  folder=ROOT/'results'/('discovery_original' if split=='discovery' else 'frozen_'+split)
  path=folder/'scores.jsonl'
  if not path.exists():
   assert not a.include_test,f'Requested final analysis is missing {split}'
   continue
  # Inspect completion receipt before reading any examples, especially test.
  expected_n={'discovery':192,'selection':192,'validation':299,'test':1172}[split]
  receipt=folder/(f'COMPLETE_{expected_n}.json' if split=='discovery' else 'COMPLETE.json')
  completion=json.loads(receipt.read_text(encoding='utf8'))
  assert completion['status']=='complete' and completion['n']==expected_n
  r=[json.loads(l) for l in path.read_text(encoding='utf8').splitlines()]
  if split=='discovery':assert (folder/f'COMPLETE_{len(r)}.json').exists()
  else:assert (folder/'COMPLETE.json').exists(),'Never summarize an incomplete evaluation as final'
  assert len({x['id'] for x in r})==len(r)==expected_n
  allrows[split]=r
  methods=list(r[0]['methods'])
  assert all(set(x['methods'])==set(methods) for x in r),'Inconsistent frozen methods across questions'
  for x in r:
   for method in methods:
    item=x['methods'][method];values=np.asarray(item['normalized_scores'],dtype=float)
    assert np.isfinite(values).all() and np.isfinite(item['margin'])
    assert int(values.argmax())==item['prediction']
    assert bool(item['correct'])==(int(item['prediction'])==int(x['gold']))
    expected_margin=values[int(x['gold'])]-np.delete(values,int(x['gold'])).max()
    assert abs(float(expected_margin)-float(item['margin']))<=1e-12
  for method in methods:
   if method.startswith('raw_chord') or method in ['normalized_chord_0','normalized_chord_1']:continue
   c=sum(x['methods'][method]['correct'] for x in r)
   summary.append(dict(split=split,method=method,k=int(method[-1]) if method.startswith('native_k') else (4 if method.startswith('loopcd') else 5),n=len(r),correct=c,accuracy=c/len(r),accuracy_percent=100*c/len(r),mean_gold_margin=float(np.mean([x['methods'][method]['margin'] for x in r])),scope='diagnostic_discovery' if split=='discovery' else 'held_out_train' if split=='selection' else 'official_validation' if split=='validation' else 'locked_regression_benchmark',fewshot=25,dtype='BF16',metric='acc_norm_character_length'))
  comparisons=[('native_k3','native_k4'),('native_k4','native_k5'),('native_k5','native_k6'),('native_k4','loopcd_k4_fixed'),('native_k4','loopcd_k4_adaptive')]
  comparisons += [('native_k5',m) for m in methods if m.startswith('normalized_chord_') and m not in ['normalized_chord_0','normalized_chord_1']]
  for before,after in comparisons:
   contrasts.append(paired(r,before,after,split))
   for x in r:
    b,c=[x['methods'][m] for m in [before,after]]
    pairs.append(dict(split=split,id=x['id'],before=before,after=after,gold=x['gold'],prediction_before=b['prediction'],prediction_after=c['prediction'],correct_before=b['correct'],correct_after=c['correct'],transition='W_to_C' if not b['correct'] and c['correct'] else 'C_to_W' if b['correct'] and not c['correct'] else 'C_to_C' if c['correct'] else 'W_to_W',margin_before=b['margin'],margin_after=c['margin'],delta_margin=c['margin']-b['margin'],normalized_scores_before=json.dumps(b['normalized_scores']),normalized_scores_after=json.dumps(c['normalized_scores']),prompt_sha256=x['prompt_sha256']))
 # Holm-adjust within each split's reported secondary family, never as a primary claim.
 for split in allrows:
  ordered=sorted([c for c in contrasts if c['split']==split],key=lambda c:c['exact_mcnemar_p']);running=0
  for i,c in enumerate(ordered):
   running=max(running,min(1.,c['exact_mcnemar_p']*(len(ordered)-i)));c['holm_adjusted_p_within_split']=running;c['holm_family_size']=len(ordered)
 writecsv(ROOT/'evaluation_summary.csv',summary);writecsv(ROOT/'paired_transitions.csv',pairs);writecsv(ROOT/'paired_statistics.csv',contrasts)
 save('results/EVALUATION_ANALYSIS.json',dict(splits={s:len(r) for s,r in allrows.items()},contrasts=contrasts,primary_weight_contrast_available=False,test_selection_performed=False))
 if summary:
  import matplotlib;matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
  fig,axs=plt.subplots(1,2,figsize=(10,3.7),layout='constrained')
  colors={'discovery':'#3B6EA8','selection':'#D67A32','validation':'#579568','test':'#8765A3'}
  for split in allrows:
   rows=[r for r in summary if r['split']==split and r['method'] in ['native_k3','native_k4','native_k5','native_k6']]
   axs[0].plot([r['k'] for r in rows],[r['accuracy_percent'] for r in rows],marker='o',label=f'{split} (n={len(allrows[split])})',color=colors[split])
  axs[0].set(xlabel='Recurrent loops K',ylabel='ARC-Challenge acc_norm (%)',xticks=[3,4,5,6]);axs[0].legend(fontsize=8)
  show=[c for c in contrasts if c['after']=='native_k5'];labels=[c['split'] for c in show];x=np.arange(len(show))
  axs[1].bar(x-.18,[c['rescues'] for c in show],.36,label='Wrong → correct',color='#3E8E78');axs[1].bar(x+.18,[c['regressions'] for c in show],.36,label='Correct → wrong',color='#C66A62')
  axs[1].set(xticks=x,xticklabels=labels,ylabel='Question count',title='Native K=4 → K=5');axs[1].legend(fontsize=8)
  (ROOT/'figures').mkdir(exist_ok=True)
  for ext in ['png','svg','pdf']:fig.savefig(ROOT/'figures'/('accuracy_and_transitions.'+ext),dpi=200)
  plt.close(fig)
 print(json.dumps(dict(splits={s:len(r) for s,r in allrows.items()},summary_rows=len(summary),paired_rows=len(pairs)),indent=2))
if __name__=='__main__':main()
