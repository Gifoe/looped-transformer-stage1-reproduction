"""Independent reduced-scope CPU audit; never imports an execution module."""
import argparse,json,hashlib,math,collections,statistics
from pathlib import Path
from fractions import Fraction
import torch
from safetensors.torch import load_file
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def am(x):return max(range(len(x)),key=lambda i:x[i])
def metrics(v,g,f):
 w=[x for i,x in enumerate(v) if i!=g];return {'correct':am(v)==g,'ties':sum(x==max(v) for x in v)>1,'meanwrong':v[g]-sum(w)/len(w),'fixedwrong':v[g]-v[f],'dynamicmargin':v[g]-max(w)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--inputs',required=True);ap.add_argument('--cohort',required=True);ap.add_argument('--labels',required=True);ap.add_argument('--expected-n',type=int,choices=[2,128],required=True);ap.add_argument('--output',required=True);a=ap.parse_args();root=Path(a.results);cohort=read(a.cohort);expected=cohort['technical_ids'] if a.expected_n==2 else cohort['ids'];expectedset=set(expected);inputs={q['id']:q for q in map(json.loads,Path(a.inputs).read_text().splitlines())}
 text=Path(a.labels).read_text()
 try:
  parsed=json.loads(text);labelrows=list(parsed.values()) if isinstance(parsed,dict) else parsed
 except json.JSONDecodeError:labelrows=list(map(json.loads,text.splitlines()))
 labels={(q['split'],q['id']):q for q in labelrows};identity=read(root/'execution_identity.json');receipt=read(root/'receipt.json');donors=read(root/'control_donor_map.json');fail=[];seen=set();summaries=collections.defaultdict(list);mathbounds=[]
 def ck(ok,msg):
  if not ok:fail.append(msg)
 def scores(row,n,tag):
  v=row['candidate_logits'];s=row['scores'];ck(len(v)==len(s)==n and all(math.isfinite(z) for z in v+s),tag+' finite/count');ck(am(v)==am(s),tag+' argmax');ck([i for i,x in enumerate(v) if x==max(v)]==[i for i,x in enumerate(s) if x==max(s)],tag+' ties');ck(max(abs((s[i]-s[0])-(v[i]-v[0])) for i in range(n))<2e-5,tag+' FP32 pair gaps');ck(max(abs(v[i]-row['logsumexp_fp32']-s[i]) for i in range(n))<2e-5,tag+' FP32 normalization')
 scope='NATIVE_K1_K4_IDENTITY_K1_K4_K2_KV_TARGET_NORM_UNRELATED_ONLY';ck(receipt['status']=='COMPLETE' and receipt['N']==a.expected_n,'receipt N/status');ck(receipt['scope']==identity['scope']==scope and receipt['unexecuted_original_six_direction_design'] is True,'explicit reduced scope');ck(receipt['identity']==identity,'receipt identity');ck(identity['inputs_sha256']==sha(a.inputs) and identity['cohort_sha256']==sha(a.cohort),'input/cohort file SHA');ck(identity['model_sha256']=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af','checkpoint');ck(identity['donor_map_sha256']==sha(root/'control_donor_map.json'),'donor SHA')
 ordered=sorted(expected);expectedmap={i:ordered[(j+1)%len(ordered)] for j,i in enumerate(ordered)};ck(donors==expectedmap,'sorted frozen donor map');ck(identity['actual_selected_ids_sha256']==hashlib.sha256(json.dumps(expected).encode()).hexdigest(),'actual selected IDs SHA')
 loaded=read(root/'loaded_source_identity.json');ck(loaded['loaded_model_source_sha256']==identity['model_source_sha256'] and loaded['loaded_config_source_sha256']==identity['configuration_sha256'],'loaded source actual/disk')
 for p in sorted(root.glob('*.results.json')):
  obj=read(p);ident=obj['id'];tag=obj['split']+'/'+ident;ck(ident not in seen and ident in expectedset,tag+' ID coverage');seen.add(ident);q=inputs[ident];gold=labels[(obj['split'],ident)]['gold'];np=p.with_name(p.name.replace('.results.json','.native.json'));m=read(np);tp=np.with_suffix('.safetensors');ip=p.with_name(p.name.replace('.results.json','.interventions.safetensors'));t=load_file(str(tp));it=load_file(str(ip));ck(sha(np)==obj['native_file_sha256'] and sha(tp)==m['tensor_sha256'] and sha(ip)==obj['interventions_tensor_sha256'],tag+' file hashes');ck(m['identity']==obj['identity']==identity and m['scope']==obj['scope']==scope,tag+' same identity/scope');ck(m['targets']==q['targets'] and m['R']==q['R']==len(q['ids'])-1 and m['input_ids_sha256']==q['input_ids_sha256']==hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest() and m['metadata_sha256']==q['metadata_sha256'],tag+' tokens/targets/R/metadata');n=len(q['targets']);ck(0<=gold<n,tag+' gold mapping');ck(set(m['native'])=={'1','2','3','4'},tag+' four native loops')
  wanted={f'K{d}_{k}' for d in range(1,5) for k in ('O','a_R','x_R','final_norm_R')}|{'2_1_target'};ck(set(t)==wanted,tag+' exact native tensor scope')
  for d in range(1,5):
   scores(m['native'][str(d)],n,tag+f' N{d}');cr=m['capture_receipts'][str(d)];ck(cr['loop_events']==list(range(d)) and cr['outside_R_unchanged'],tag+f' actual loop trace{d}');shape=cr['sdpa_args']['q']['shape'];ck(shape[-2]==len(q['ids']),tag+' complete SDPA sequence')
   for k in ('O','a_R','x_R','final_norm_R'):ck(t[f'K{d}_{k}'].dtype==torch.bfloat16 and bool(torch.isfinite(t[f'K{d}_{k}']).all()),tag+' BF16 finite states')
  grid=[(x['receiver_loop'],x['donor_loop'],x['condition']) for x in obj['rows']];required={(d,d,'identity') for d in range(1,5)}|{(2,1,c) for c in ('target','norm_only','unrelated_delta')};ck(len(grid)==len(set(grid))==7 and set(grid)==required,tag+' exact 4identity+3KV grid')
  for row in obj['rows']:
   scores(row,n,tag+' '+row['condition']);ck(row['outside_R_unchanged'],tag+' outside R');d=row['receiver_loop'];base=m['native'][str(d)]
   if row['condition']=='identity':ck(row['full_logits_sha256']==base['full_logits_sha256'],tag+f' exact identity{d}');continue
   c=row['condition'];ck(row['native']['full_logits_sha256']==base['full_logits_sha256'] and row['pair']=='1-2' and row['direction']=='rollback',tag+' current recipient');ck((row['donor_id']==donors[ident] if not row.get('donor_fallbacks') else row['donor_id']!=ident) if c=='unrelated_delta' else row['donor_id']==ident,tag+' donor')
   if c=='unrelated_delta':
    current=donors[ident]
    for skipped in row.get('donor_fallbacks',[]):
     ck(skipped==current,tag+' frozen fallback sequence');ks='train_'+hashlib.sha256(skipped.encode()).hexdigest();z=load_file(str(root/(ks+'.native.safetensors')));ck(torch.equal(z['2_1_target'],z['K2_O']),tag+' fallback source actually zero');current=donors[current]
    ck(row['donor_id']==current and current!=ident,tag+' exact final donor')
   actual=it[c+'_injected'];ck(thash(actual)==row['injected_sha256'] and thash(t['K2_O'])==row['native_sdpa_R_sha256'] and thash(it[c+'_a_R'])==row['a_R_sha256'],tag+' injected/source hashes')
   if c=='target':ck(thash(actual)==thash(t['2_1_target']),tag+' exact actual KV target')
   else:
    b=t['K2_O'].double();tar=t['2_1_target'].double()
    if c=='norm_only':ideal=b*(tar.norm()/b.norm())
    else:
     di=row['donor_id'];ds='train_'+hashlib.sha256(di.encode()).hexdigest();dt=load_file(str(root/(ds+'.native.safetensors')));od=dt['2_1_target'].double()-dt['K2_O'].double();ck(float(od.norm())>0,tag+' nonzero donor');ideal=b+(tar-b).norm()*od/od.norm()
    # Standard floating-point gamma bound for two FP32 norms and elementwise operations,
    # plus nearest BF16 conversion error. No empirical threshold is selected from outcomes.
    count=b.numel();u=2**-24;gamma=((8*count+32)*u)/(1-(8*count+32)*u);bound=gamma*(b.abs()+(ideal-b).abs())+ideal.abs()*2**-8+2**-134;err=(actual.double()-ideal).abs();ck(bool((err<=bound).all()),tag+' independently bounded control arithmetic');mathbounds.append({'id':ident,'condition':c,'max_abs_error_vs_FP64_ideal':float(err.max()),'max_FP32_plus_BF16_bound':float(bound.max()),'inside_all_component_bounds':bool((err<=bound).all())})
   fixed=max((i for i in range(n) if i!=gold),key=lambda i:base['scores'][i]);glob=max((i for i in range(n) if i!=gold),key=lambda i:m['native']['1']['scores'][i]);bv=metrics(base['scores'],gold,fixed);iv=metrics(row['scores'],gold,fixed);ex=metrics(list(map(Fraction,row['candidate_logits'])),gold,fixed);eb=metrics(list(map(Fraction,base['candidate_logits'])),gold,fixed);rec={'correct':int(iv['correct']),'ties':int(iv['ties']),'rescue':int(not bv['correct'] and iv['correct']),'damage':int(bv['correct'] and not iv['correct'])}
   for k in ('meanwrong','fixedwrong','dynamicmargin'):rec[k+'_delta']=iv[k]-bv[k];rec[k+'_exact_sign']=int(ex[k]>eb[k])-int(ex[k]<eb[k])
   rec['global_K1_fixedwrong_delta']=metrics(row['scores'],gold,glob)['fixedwrong']-metrics(base['scores'],gold,glob)['fixedwrong'];summaries[c].append(rec)
 ck(seen==expectedset,'exact requested-ID coverage');summary=[]
 for c,rows in sorted(summaries.items()):
  s={'condition':c,'n':len(rows)}
  for k in ('correct','ties','rescue','damage'):s[k]=sum(x[k] for x in rows)
  for k in ('meanwrong_delta','fixedwrong_delta','dynamicmargin_delta','global_K1_fixedwrong_delta'):s[k]={'mean':statistics.mean(x[k] for x in rows),'median':statistics.median(x[k] for x in rows)}
  summary.append(s)
 result={'status':'FAILED' if fail else ('PASS_COMPLETE' if a.expected_n==len(cohort['ids']) else 'PASS_PARTIAL'),'n_expected':len(cohort['ids']),'n_observed':len(seen),'requested_scope_n':a.expected_n,'scope':scope,'failures':fail,'summary':summary,'control_math_bounds':mathbounds,'producer_imports':False};Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','n_expected','n_observed','failures')}));raise SystemExit(bool(fail))
if __name__=='__main__':main()
