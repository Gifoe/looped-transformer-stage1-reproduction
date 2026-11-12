"""Independent four-corner state/attention audit and exact outcome accounting."""
import argparse,json,hashlib,collections,statistics
from pathlib import Path
from fractions import Fraction
import torch
from safetensors.torch import load_file
from astra_independent_audit import metrics,argmax,group
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.view(torch.uint8),b.view(torch.uint8))
def thash(x):return hashlib.sha256(x.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--receiver',type=int,required=True);p.add_argument('--results',required=True);p.add_argument('--prior',required=True);p.add_argument('--labels',required=True);p.add_argument('--cohort',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 depth=a.receiver;donor_depth=depth-1;assert depth in (3,4);root=Path(a.results);prior=Path(a.prior);labels={(r['split'],r['id']):r for r in map(json.loads,Path(a.labels).read_text(encoding='utf-8-sig').splitlines())};expected=set(read(a.cohort)['ids']);seen=set();fail=[];buckets=collections.defaultdict(list);contrasts=[];cases=[];precision=[];au_a1_equal=0
 def check(ok,msg):
  if not ok:fail.append(msg)
 def ties(v):return sum(x==max(v) for x in v)>1
 native_by_id={read(path)['id']:path for path in root.glob('*.native.json')}
 control_math_rounding=[]
 def scorecheck(r,tag):
  v,s=r['candidate_logits'],r['scores'];check(max(abs((s[i]-s[0])-(v[i]-v[0])) for i in range(len(v)))<2e-5,tag+' scores');check(argmax(v)==argmax(s) and ties(v)==ties(s),tag+' prediction/ties')
 for path in sorted(root.glob('*.results.json')):
  obj=read(path);ident=obj['id'];tag=obj['split']+'/'+ident;check(ident not in seen,tag+' duplicate');seen.add(ident);g=labels[(obj['split'],ident)]['gold'];np=path.with_name(path.name.replace('.results.json','.native.json'));m=read(np);op=prior/np.name;old=read(op)
  check(m['receiver_loop']==depth and m['donor_loop']==donor_depth,tag+' explicit depth')
  check(m['identity']['precision_ids']==sorted(expected)[:8] if seen==expected else set(m['identity']['precision_ids'])<=set(sorted(expected)[:8]),tag+' precision selection')
  check(sha(np)==obj['native_file_sha256'] and sha(op)==m['prior_native_sha256'],tag+' metadata hash');check(sha(op.with_suffix('.safetensors'))==old['tensor_sha256']==m['prior_tensor_sha256'],tag+' old tensor hash');check(m['input_ids_sha256']==old['input_ids_sha256'],tag+' input identity')
  check(sha(np.with_suffix('.safetensors'))==m['tensor_sha256'],tag+' tensor hash');ip=path.with_name(path.name.replace('.results.json','.interventions.safetensors'));check(sha(ip)==obj['interventions_tensor_sha256'],tag+' intervention tensor hash')
  t=load_file(str(np.with_suffix('.safetensors')));it=load_file(str(ip));ot=load_file(str(op.with_suffix('.safetensors')));n=m['native'][str(depth)];first=m['reused_K1_native'];adjacent=m['reused_donor_native'];check(adjacent==old['native'][str(donor_depth)] and first==old['native']['1'],tag+' adjacent/global baseline identity');check(n['full_logits_sha256']==old['native'][str(depth)]['full_logits_sha256'],tag+' native bridge')
  for k in ['x_R','O','a_R','final_norm_R']:check(bits(t['N_'+k],ot['K'+str(depth)+'_'+k]),tag+' native tensor '+k)
  check(bits(t['X1'],ot['K'+str(donor_depth)+'_x_R']) and bits(t['U_x_R'],t['X1']),tag+' X1 source');check(bits(t['A1'],ot['K'+str(donor_depth)+'_O']),tag+' A1 source');au_a1_equal+=int(bits(t['U_O'],t['A1']))
  rows={r['condition']:r for r in obj['rows']};check(len(rows)==len(obj['rows']),tag+' duplicate condition');required={'U','identity','U_restore','A_only','state_norm_only','state_unrelated_delta','mediator_norm_only','mediator_unrelated_delta'};check(required-set(obj.get('unavailable',{}))<=set(rows),tag+' condition coverage')
  check(rows['identity']['full_logits_sha256']==n['full_logits_sha256'],tag+' identity');check(rows['U']['full_logits_sha256']==m['upstream_block']['full_logits_sha256'],tag+' actual U endpoint')
  if 'N' in rows:check(rows['N']['full_logits_sha256']==n['full_logits_sha256'],tag+' explicit N cell')
  for name,x,o,injected in [('identity',t['N_x_R'],t['N_O'],t['N_O']),('U_restore',t['X1'],t['U_O'],t['N_O']),('A_only',t['N_x_R'],t['N_O'],t['U_O'])]:
   check(bits(it[name+'_x_R'],x) and bits(it[name+'_O'],o) and bits(it[name+'_injected'],injected),tag+' actual four-corner '+name);check(thash(it[name+'_injected'])==rows[name]['injected_sha256'],tag+' injected hash '+name)
  check(bits(it['U_restore_a_R'],t['N_a_R']) and bits(it['A_only_a_R'],t['U_a_R']),tag+' normalized mediator bridge')
  for prefix in ['N','U']:check(bits(t[prefix+'_after_attention_R'],t[prefix+'_x_R']+t[prefix+'_a_R']),tag+' residual addition '+prefix)
  for prefix in ['identity','U_restore','A_only']:check(bits(it[prefix+'_after_attention_R'],it[prefix+'_x_R']+it[prefix+'_a_R']),tag+' residual addition '+prefix)
  for stem,base,target,field in [('state',t['N_x_R'],t['X1'],'x_R'),('mediator',t['U_O'],t['N_O'],'injected')]:
   name=stem+'_norm_only'
   if name in rows:
    expected_norm=(base.float()*(target.float().norm()/base.float().norm())).to(base.dtype)
    check(bits(it[name+'_'+field],expected_norm),tag+' control norm math '+stem)
   name=stem+'_unrelated_delta'
   if name in rows:
    donor=rows[name]['donor_id'];donor_paths=[native_by_id[donor]] if donor in native_by_id else [];check(len(donor_paths)==1,tag+' donor source '+stem)
    if len(donor_paths)==1:
     dt=load_file(str(donor_paths[0].with_suffix('.safetensors')))
     dd=dt['X1'].float()-dt['N_x_R'].float() if stem=='state' else dt['N_O'].float()-dt['U_O'].float()
     expected_unrelated=(base.float()+(target.float()-base.float()).norm()*dd/dd.norm()).to(base.dtype)
     actual=it[name+'_'+field]
     if not bits(actual,expected_unrelated):
      dd64=dt['X1'].double()-dt['N_x_R'].double() if stem=='state' else dt['N_O'].double()-dt['U_O'].double()
      ideal=base.double()+(target.double()-base.double()).norm()*dd64/dd64.norm()
      nearest=ideal.to(base.dtype);up=torch.nextafter(nearest,torch.full_like(nearest,float('inf'))).double();down=torch.nextafter(nearest,torch.full_like(nearest,-float('inf'))).double()
      rounding_bound=torch.maximum((up-nearest.double()).abs(),(nearest.double()-down).abs())/2
      fp32_allowance=2e-6*max(float(base.double().abs().max()),float((target.double()-base.double()).norm()),1e-12)
      valid=bool(((actual.double()-ideal).abs()<=rounding_bound+fp32_allowance).all())
      control_math_rounding.append({'id':ident,'condition':name,'different_elements':int((actual!=expected_unrelated).sum()),'max_difference_from_local_FP32':float((actual.float()-expected_unrelated.float()).abs().max()),'FP64_rounding_bound_pass':valid,'fp32_allowance':fp32_allowance,'target_delta_norm_fp64':float((target.double()-base.double()).norm()),'actual_delta_norm_fp64':float((actual.double()-base.double()).norm())})
      check(valid,tag+' control unrelated FP64 rounding bound '+stem)
  fixed=max((i for i in range(len(n['scores'])) if i!=g),key=lambda i:n['scores'][i]);globalfixed=max((i for i in range(len(n['scores'])) if i!=g),key=lambda i:first['scores'][i]);bm=metrics(n['scores'],g,fixed);ng=group(argmax(adjacent['scores'])==g,bm['correct']);natnotie=not(ties(adjacent['scores'])or ties(n['scores']));scorecheck(n,tag+' N')
  for name,r in rows.items():
   check(r['receiver_loop']==depth and r['donor_loop']==donor_depth and r['pair']==str(donor_depth)+'-'+str(depth),tag+' row depth')
   scorecheck(r,tag+' '+name);check(r['outside_R_unchanged'],tag+' outsideR')
   for k in ['non_R_x_sha256','non_R_Q_sha256','non_R_K_sha256','non_R_V_sha256','non_R_sdpa_sha256']:check(r['site_receipt'][k]==m['N_site_receipt'][k],tag+' '+name+' '+k)
   if name.endswith('unrelated_delta'):check(r['donor_id']!=ident,tag+' unrelated donor')
   if name in ['U_restore','mediator_norm_only','mediator_unrelated_delta']:
    for k in ['Q_R','K_R','V_R']:check(r['site_receipt'][k]['sha256']==m['U_site_receipt'][k]['sha256'],tag+' upstream retained '+k)
   im=metrics(r['scores'],g,fixed);rec={'id':ident,'condition':name,'group':ng,'correct':im['correct'],'rescue':int(not bm['correct'] and im['correct']),'damage':int(bm['correct'] and not im['correct']),'ties':im['ties']}
   for met in ['meanwrong','fixedwrong','dynamicmargin']:rec[met+'_delta']=im[met]-bm[met]
   rec['global_fixedwrong_delta']=metrics(r['scores'],g,globalfixed)['fixedwrong']-metrics(n['scores'],g,globalfixed)['fixedwrong']
   for strat in ['all',ng]:buckets[(obj['split'],name,strat)].append(rec)
  vals={name:metrics(r['scores'],g,fixed) for name,r in {'N':n,**rows}.items()};exact={name:metrics([Fraction(x) for x in r['candidate_logits']],g,fixed) for name,r in {'N':n,**rows}.items()}
  for name,r in {'N':n,**rows}.items():
   vals[name]['global_fixedwrong']=metrics(r['scores'],g,globalfixed)['fixedwrong']
   exact[name]['global_fixedwrong']=metrics([Fraction(x) for x in r['candidate_logits']],g,globalfixed)['fixedwrong']
  cr={'id':ident,'split':obj['split'],'group':ng}
  specs={'total':{'U':1,'N':-1},'attention_only':{'A_only':1,'N':-1},'direct_at_native_attention':{'U_restore':1,'N':-1},'restore_effect':{'U_restore':1,'U':-1},'interaction':{'U':1,'U_restore':-1,'A_only':-1,'N':1},'norm_restore_effect':{'mediator_norm_only':1,'U':-1},'unrelated_restore_effect':{'mediator_unrelated_delta':1,'U':-1}}
  for contrast,weights in specs.items():
   if not set(weights)<=set(vals):continue
   cr[contrast]={}
   for met in ['meanwrong','fixedwrong','dynamicmargin','global_fixedwrong']:
    ex=sum(w*exact[k][met] for k,w in weights.items());cr[contrast][met]={'value':sum(w*vals[k][met] for k,w in weights.items()),'exact_raw':float(ex),'exact_sign':(ex>0)-(ex<0)}
  contrasts.append(cr)
  if bm['correct'] and not vals['U']['correct']:
   cases.append({'id':ident,'group':ng,'native_no_tie':natnotie,'U_tie':bool(vals['U']['ties']),'restore_correct':bool(vals['U_restore']['correct']),'restore_tie':bool(vals['U_restore']['ties']),'A_only_correct':bool(vals['A_only']['correct']),'state_norm_correct':bool(vals.get('state_norm_only',{}).get('correct')),'state_unrelated_correct':bool(vals.get('state_unrelated_delta',{}).get('correct')),'mediator_norm_correct':bool(vals.get('mediator_norm_only',{}).get('correct')),'mediator_unrelated_correct':bool(vals.get('mediator_unrelated_delta',{}).get('correct')),'N_logits':n['candidate_logits'],'U_logits':rows['U']['candidate_logits'],'U_restore_logits':rows['U_restore']['candidate_logits']})
  if 'U_fp32_sdpa_bf16_writeback' in rows:precision.append({'id':ident,'prediction_changed':argmax(rows['U']['scores'])!=argmax(rows['U_fp32_sdpa_bf16_writeback']['scores'])})
 check(seen<=expected,'unexpected IDs');ids=sorted(seen)
 for name in ['state_donor_map.json','mediator_donor_map.json']:
  dm=read(root/name);check(set(dm)==seen and all(dm[x]==ids[(i+1)%len(ids)] for i,x in enumerate(ids)),name+' sorted map')
 summary=[]
 for key,rs in sorted(buckets.items()):
  z=dict(zip(['split','condition','stratum'],key));z['n']=len(rs)
  for k in ['correct','rescue','damage','ties']:z[k]=sum(r[k] for r in rs)
  for k in ['meanwrong_delta','fixedwrong_delta','dynamicmargin_delta','global_fixedwrong_delta']:z[k]={'mean':statistics.mean(r[k] for r in rs),'median':statistics.median(r[k] for r in rs)}
  summary.append(z)
 cs=[]
 for ng in ['all','rescue','damage','stable_correct','stable_wrong']:
  rr=[r for r in contrasts if ng=='all' or r['group']==ng]
  for cn in specs:
   rr2=[r for r in rr if cn in r]
   if not rr2:continue
   z={'group':ng,'contrast':cn,'n':len(rr2)}
   for met in ['meanwrong','fixedwrong','dynamicmargin','global_fixedwrong']:z[met]={'mean':statistics.mean(r[cn][met]['value'] for r in rr2),'exact_positive':sum(r[cn][met]['exact_sign']>0 for r in rr2),'exact_negative':sum(r[cn][met]['exact_sign']<0 for r in rr2),'exact_zero':sum(r[cn][met]['exact_sign']==0 for r in rr2)}
   cs.append(z)
 result={'status':'FAILED' if fail else ('PASS_COMPLETE' if seen==expected else 'PASS_PARTIAL'),'n_expected':len(expected),'n_observed':len(seen),'failures':fail,'control_math_cross_cpu_rounding':control_math_rounding,'receiver_loop':depth,'donor_loop':donor_depth,'A_U_equals_preceding_native_A_count':au_a1_equal,'summary':summary,'contrasts':cs,'contrast_rows':contrasts,'upstream_damage_cases':cases,'fp32_sdpa_sensitivity':precision}
 Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:result[k] for k in ['status','n_expected','n_observed','failures']}))
 if fail:raise SystemExit(1)
if __name__=='__main__':main()
