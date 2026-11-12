"""Generate factual factorial and counterexample reports from independently audited rows."""
import json,gzip,hashlib,datetime,os
from pathlib import Path
B=Path('/home/luansiyu/loop-latent/research_plans/natural_qkv_eight_corner_20261112')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
N=[f'G{i}{j}{k}' for i in (1,2) for j in (1,2) for k in (1,2)]
E={**{f'Q_K{j}V{k}':(f'G1{j}{k}',f'G2{j}{k}') for j in (1,2) for k in (1,2)},**{f'K_Q{i}V{k}':(f'G{i}1{k}',f'G{i}2{k}') for i in (1,2) for k in (1,2)},**{f'V_Q{i}K{j}':(f'G{i}{j}1',f'G{i}{j}2') for i in (1,2) for j in (1,2)}}
head=['# Round 1 factorial analysis','', 'Within-project train-informed prospective comparison; earlier ARC and four-corner outcomes were exposed. F is the original production-score gold versus all non-gold mean within question, then equally averaged across questions. K and V switch full sequence; Q only R. The interventions are mixed states at one fixed receiver, not a unique natural causal decomposition.','', 'I_Qi=Gi22−Gi21−Gi12+Gi11; J=I_Q2−I_Q1. Fraction signs use exact BF16 target logits, with no epsilon. Dynamic gap and per-candidate scores remain available in complete rows and ROUND1_FULL_RESULTS.json.','', 'Each group table reports all 12 edges and interactions; the exact full distributions, fixed-wrong sensitivity, vector norms and subgroup details are bound in each summary JSON. All eight corners used full-vocabulary CUDA score replay and all four prior corners were checked bitwise against prior QKV. In-forward attribution rests on the audited frozen collector; deleted original natural whole shards were inherited from historical coverage, not reread.','']
contra=['# Round 1 counterexamples','', 'All listed IDs are exhaustive within each stated rule and split, sorted lexicographically. For each edge a→b, a damage example changes correct→incorrect even if its score gap rises. Joint-only means G_i22 correct while G_i11, G_i12, G_i21 are all wrong. All candidates, ties and every raw score are retained in ROUND1_FULL_RESULTS.json and split rows.','']
for split in ('train','validation','test'):
 base=B/(split+'_analysis');summary=base/'summary.json';cov=B/(split+'_independent_cuda_coverage.json');rows=base/'rows.jsonl.gz';s=json.loads(summary.read_text());assert s['coverage_sha256']==sha(cov) and s['rows_sha256']==sha(rows)
 with gzip.open(rows,'rt') as f:rs=[json.loads(x) for x in f]
 assert len(rs)==s['count'] and len({x['id'] for x in rs})==s['count']
 head.extend([f'## {split} (N={s["count"]})','',f'Coverage SHA256 `{sha(cov)}`; summary SHA256 `{sha(summary)}`; rows SHA256 `{sha(rows)}`.',''])
 for name,g in s['groups'].items():
  if not g['N']:continue
  head.extend([f'### {name} (N={g["N"]})','','| Contrast | Mean | Median | 5th–95th | Production +/−/0 | Fraction +/−/0 | Fixed-wrong mean |','|---|---:|---:|---:|---:|---:|---:|'])
  for k,d in g['gap_differences'].items():
   f=g['fraction_signs'][k];w=g['fixed_differences'][k];head.append(f'| {k} | {d["mean"]:.6g} | {d["median"]:.6g} | {d["q05"]:.6g} to {d["q95"]:.6g} | {d["positive"]}/{d["negative"]}/{d["zero"]} | {f["positive"]}/{f["negative"]}/{f["zero"]} | {w["mean"]:.6g} |')
  head.append('')
  head.append('Correct counts: '+', '.join(f'{k}={v}' for k,v in sorted(g['correct'].items()))+'.')
  head.append('Edge rescue/damage/net: '+', '.join(f'{k}={v["rescue"]}/{v["damage"]}/{v["net"]}' for k,v in sorted(g['edges'].items()))+'.\n')
 if split!='train':
  p=base/'prospective_checks.json';v=json.loads(p.read_text());assert v['summary_sha256']==sha(summary) and v['coverage_sha256']==sha(cov) and v['rows_sha256']==sha(rows);head.extend(['### Frozen prospective checks','',f'Full individual pass/fail JSON: `{p.relative_to(B)}` (SHA256 `{sha(p)}`).',''])
 contra.extend([f'## {split}','','| Rule | ID | Gold | Corner predictions | Corner all-wrong gaps |','|---|---|---:|---|---|'])
 for name,(a,b) in E.items():
  selected=sorted((r for r in rs if r['predictions'][a]==r['gold'] and r['predictions'][b]!=r['gold']),key=lambda r:r['id'])
  for r in selected:contra.append(f'| {name} damage | {r["id"]} | {r["gold"]} | {a}:{r["predictions"][a]}, {b}:{r["predictions"][b]} | {a}:{r["score_gap"][a]:.7g}, {b}:{r["score_gap"][b]:.7g} |')
 for i in (1,2):
  cs=[f'G{i}11',f'G{i}12',f'G{i}21'];joint=f'G{i}22'
  for r in sorted((r for r in rs if all(r['predictions'][k]!=r['gold'] for k in cs) and r['predictions'][joint]==r['gold']),key=lambda r:r['id']):contra.append(f'| Q{i} joint-only | {r["id"]} | {r["gold"]} | '+', '.join(f'{k}:{r["predictions"][k]}' for k in cs+[joint])+' | '+', '.join(f'{k}:{r["score_gap"][k]:.7g}' for k in cs+[joint])+' |')
 contra.append('')
for name,lines in [('ROUND1_FACTORIAL_ANALYSIS.md',head),('ROUND1_COUNTEREXAMPLES.md',contra)]:
 p=B/name
 with p.open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n');f.flush();os.fsync(f.fileno())
 print(name,sha(p),flush=True)
