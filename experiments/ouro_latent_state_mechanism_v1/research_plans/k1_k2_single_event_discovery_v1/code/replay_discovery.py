"""Portable stdlib-only independent Fraction replay of every selection candidate."""
import pathlib,json,gzip,hashlib
from fractions import Fraction as F
R=pathlib.Path(__file__).resolve().parent.parent
mp=json.loads((R/'SHARDED_ARTIFACTS.json').read_text())
def data(name):
 z=next((z for z in mp['artifacts'] if z['path']==name),None)
 b=b''.join((R/p).read_bytes() for p in z['parts']) if z else (R/name).read_bytes()
 if z:assert len(b)==z['bytes'] and hashlib.sha256(b).hexdigest()==z['sha256']
 return gzip.decompress(b).decode()
rows=[json.loads(l) for l in data('DISCOVERY_ROWS.jsonl.gz').splitlines()];qs={q['id']:q for q in map(json.loads,data('DEV_INPUTS.jsonl.gz').splitlines())};assert len(qs)==512 and len(rows)==4096 and len({(u['id'],u['layer']) for u in rows})==4096
s=json.loads((R/'LAYER_SELECTION.json').read_text());mins={};eligible={}
def y(u,k):
 x=list(map(F,u[k]['scores']['2']['candidate_logits']));return x[u['opposite_index']]-(x[u['decoy_indices'][0]]+x[u['decoy_indices'][1]])/2
def mean(a):return sum(a,F(0))/len(a)
for layer in range(9,17):
 rr=[u for u in rows if u['layer']==layer];assert {u['id'] for u in rr}==set(qs);p=s['candidate_profiles'][str(layer)];four={f'F{f}_E{e}':mean([y(u,'semantic')-y(u,'random') for u in rr if (u['firsthop_factor'],u['endpoint_factor'])==(f,e)]) for f in [0,1] for e in [0,1]};fine={f'F{f}_E{e}_Q{q}':mean([y(u,'semantic')-y(u,'random') for u in rr if (u['firsthop_factor'],u['endpoint_factor'],u['selector'])==(f,e,q)]) for f in [0,1] for e in [0,1] for q in [0,1]};assert {k:float(v) for k,v in four.items()}==p['four_strata'];assert {k:float(v) for k,v in fine.items()}==p['eight_selector_strata'];assert float(mean([y(u,'semantic')-y(u,'native') for u in rr]))==p['mean_semantic_vs_native_Y'];mins[layer]=min(four.values());assert float(mins[layer])==p['m_minimum_stratum_effect'];allpos=all(v>0 for v in fine.values());assert allpos==p['all_eight_positive'];target=[];actual=[];match=[]
 for u in rr:
  d=u['random']['audit']['events'][0]['dose'];assert d['target']==u['semantic']['audit']['events'][0]['actual_delta_norms'];target+=d['target'];actual+=d['actual'];match+=d['matched']
 check=[abs(a-t)<=max(1e-9,.01*t) for t,a in zip(target,actual)];assert check==match;q=p['dose_QC'];er=sum(abs(a-t) for t,a in zip(target,actual))/sum(target);mass=sum(t for t,ok in zip(target,check) if ok)/sum(target);assert abs(er-q['norm_error_ratio'])<1e-12 and abs(mass-q['matched_target_mass_fraction'])<1e-12 and sum(check)==q['matched'] and len(check)==q['rows'];qc=er<=.01 and mass>=.99;assert qc==q['pass'];eligible[layer]=mins[layer]>F(1,10) and allpos and qc;assert eligible[layer]==p['eligible']
best=max(mins,key=lambda l:(mins[l],-l));assert s['maximum_minimum_candidate']==best and s['selected_layer']==(best if eligible[best] else None);assert not s['goal_complete'] and s['no_new_confirmation_run'];print('CPU_REPLAY_PASS all8profiles/4096rows/512worlds exactFraction; selected',best)
