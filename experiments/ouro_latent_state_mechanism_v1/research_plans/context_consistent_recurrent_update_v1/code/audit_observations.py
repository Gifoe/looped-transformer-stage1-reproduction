"""CPU recomputation from retained actual BF16 attention increments; no GPU/model."""
import pathlib,json,torch,datetime
R=pathlib.Path('/root/ouro-context-update-sol-20261117');torch.set_num_threads(4);cases=sites=0;maxerr=0.
for p in sorted((R/'results').glob('*.json')):
 q=json.loads(p.read_text());obs=torch.load(p.with_suffix('.pt'),map_location='cpu',weights_only=False);base=obs['N'];assert set(base)=={f'{d}:{l}' for d in [2,3,4] for l in range(10,17)}
 for name,v in obs.items():
  assert set(v)==set(base)
  if name=='N':continue
  for k,x in v.items():
   a=x.double();b=base[k].double();delta=float((a-b).norm());cos=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30));expected=q['conditions'][name]['attention_change_vs_native'][k];err=abs(delta-expected['delta_L2']);assert err<=1e-8 and abs(cos-expected['cosine'])<=1e-10;maxerr=max(err,maxerr);sites+=1
 cases+=1
assert cases==768
out={'status':'PASS','questions':cases,'attention_vector_comparisons':sites,'maximum_abs_delta_norm_error':maxerr,'scope':'Every retained BF16 L10-16 R attention vector atK2/K3/K4; CPUFP64norm/cosine recomputation vs captured scalars','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};(R/'ATTENTION_CPU_AUDIT.json').write_text(json.dumps(out,indent=2));print('ATTENTION_CPU_AUDIT_PASS',cases,sites)
