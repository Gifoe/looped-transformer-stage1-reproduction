import csv, gzip, hashlib, json
from pathlib import Path
P=Path(__file__).resolve().parent
base=P/'sources'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with (gzip.open(p,'rt',encoding='utf-8') if p.suffix=='.gz' else p.open(encoding='utf-8')) as f: return json.load(f)
sources=[]; geometry=base/'geometry-summary.json.gz'; g=read(geometry);sources.append(geometry)
with (P/'layer_curves.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f,lineterminator='\n'); w.writerow(['split','population','n','layer_ordinal_1_based','competition_k2_minus_k1','attention_k2_minus_k1','mlp_k2_minus_k1'])
    for split in ('train','validation','test'):
        group='non_demo' if split=='train' else 'all'; q=base/f'branch-{split}-summary.json.gz'; b=read(q);sources.append(q)
        gl=g['splits'][split][group]['layers']; bl=b['groups'][group]['layers'];assert len(gl)==len(bl)==24
        for i,(a,c) in enumerate(zip(gl,bl)):
            assert a['layer']==i and c['layer']==i+1 and g['splits'][split][group]['n']==c['n']
            w.writerow([split,group,c['n'],i+1,a['proxy_advantage_question_equal_weight']['mean'],c['attention_delta_difference_all_wrong']['mean'],c['mlp_delta_difference_all_wrong']['mean']])
with (P/'l18_swaps.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['split','population','n','condition','baseline_correct','swap_correct','wrong_to_correct','correct_to_wrong','net_correct'])
    for split in ('train','validation','test'):
        q=base/f'l18-{split}-summary.json'; sources.append(q); d=read(q)['groups']['non_demo' if split=='train' else 'all']
        for b,t,transition in [('B1','T12','B1_to_T12'),('B2','T21','B2_to_T21')]:
            a=d[b+'_accuracy'];z=d[t+'_accuracy'];v=d[transition];assert a['n']==z['n']==v['n'] and z['correct']-a['correct']==v['net_correct']
            w.writerow([split,'non_demo' if split=='train' else 'all',a['n'],t,a['correct'],z['correct'],v['wrong_to_correct'],v['correct_to_wrong'],v['net_correct']])
with (P/'source_sha256.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f,lineterminator='\n');w.writerow(['source_basename','sha256']);w.writerows([(str(x.relative_to(base)),digest(x)) for x in sources])
