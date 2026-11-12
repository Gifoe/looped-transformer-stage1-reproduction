import json,hashlib,pathlib,random,math,statistics
from fractions import Fraction
p=pathlib.Path(r'D:\ouro-e4-e5-20261112\branch_reference\conditional_l_holdout_snapshot');s=json.loads((p/'summary.json').read_text())['train'];rows=[r for r in (json.loads(l) for l in (p/'rows.jsonl').read_text().splitlines()) if r['split']=='train'];assert len(rows)==1030 and len({r['id'] for r in rows})==1030
pairs={'rest':('P_plus_L_main','P'),'loss':('N','N_minus_L_main'),'rest_norm':('P_plus_L_norm','P'),'loss_norm':('N','N_minus_L_norm'),'rest_unrelated':('P_plus_L_unrelated','P'),'loss_unrelated':('N','N_minus_L_unrelated')};claims=['rest','loss','rest_norm','loss_norm','rest_specific','loss_specific'];maxerr=0; exact_mismatch=0
for r in rows:
 g=r['gold'];c=r['conditions'];p1=r['native_K1_prediction'];p2=r['native_K2_prediction'];group='rescue' if p1!=g and p2==g else 'damage' if p1==g and p2!=g else 'stable_correct' if p1==g else 'stable_wrong';assert group==r['group'];vals={};raw={}
 for k,v in c.items():
  scores=v['scores']; ties=[i for i,x in enumerate(scores) if x==max(scores)];assert ties==v['ties'] and ties[0]==v['prediction'];vals[k]=scores[g]-sum(x for i,x in enumerate(scores) if i!=g)/(len(scores)-1);z=[Fraction(x) for x in v['candidate_logits']];raw[k]=z[g]-sum((x for i,x in enumerate(z) if i!=g),Fraction())/(len(z)-1)
 d={k:vals[a]-vals[b] for k,(a,b) in pairs.items()};f={k:raw[a]-raw[b] for k,(a,b) in pairs.items()};d['rest_specific']=d['rest']-d['rest_unrelated'];d['loss_specific']=d['loss']-d['loss_unrelated'];f['rest_specific']=f['rest']-f['rest_unrelated'];f['loss_specific']=f['loss']-f['loss_unrelated']
 for k in d:
  maxerr=max(maxerr,abs(d[k]-r['contrasts'][k]));exact_mismatch+=int(f[k]!=Fraction(r['fraction_contrasts'][k]))
assert maxerr<1e-12 and exact_mismatch==0
res=[r for r in rows if r['group']=='rescue'];nt=[r for r in res if r['no_tie']];rng=random.Random(20261112);draws=[[rng.randrange(len(res)) for x in res] for y in range(5000)];out={}
for k in claims:
 v=[r['contrasts'][k] for r in res];nv=[r['contrasts'][k] for r in nt];arr=sorted(sum(v[i] for i in z)/len(v) for z in draws)
 def quant(q):
  ix=(len(arr)-1)*q;j=int(ix);f=ix-j;return arr[j]*(1-f)+arr[min(j+1,len(arr)-1)]*f
 z=s['claims'][k];lo=quant(.025);hi=quant(.975);assert abs(lo-z['bootstrap_low'])<1e-12 and abs(hi-z['bootstrap_high'])<1e-12 and abs(sum(v)/len(v)-z['mean'])<1e-12 and abs(sum(nv)/len(nv)-z['no_tie_mean'])<1e-12
 out[k]={'mean':sum(v)/len(v),'lo':lo,'hi':hi,'no_tie_mean':sum(nv)/len(nv),'fraction_exact_mean':str(sum((Fraction(r['fraction_contrasts'][k]) for r in res),Fraction())/len(res))};assert out[k]['fraction_exact_mean']==z['fraction_exact_mean']
print(json.dumps({'status':'INDEPENDENT_L_TRAIN_SAVED_SCORES_AND_FRACTIONS_CHECK_PASSED','n':len(rows),'rescue':len(res),'no_tie':len(nt),'maxerr':maxerr,'fraction_mismatch':exact_mismatch,'claims':out},indent=2))
