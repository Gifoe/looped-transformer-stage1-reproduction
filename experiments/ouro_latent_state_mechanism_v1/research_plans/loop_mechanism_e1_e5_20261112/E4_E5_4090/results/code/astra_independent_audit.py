"""Independent CPU audit of Sol rows; no imports from execution/analysis code."""
import argparse, collections, hashlib, json, math, statistics
from pathlib import Path
from fractions import Fraction

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def argmax(v): return max(range(len(v)), key=lambda i:v[i])
def metrics(v,g,f):
    w=[x for i,x in enumerate(v) if i!=g]
    return {'correct':int(argmax(v)==g),'ties':int(sum(x==max(v) for x in v)>1),
            'meanwrong':v[g]-sum(w)/len(w),'fixedwrong':v[g]-v[f],'dynamicmargin':v[g]-max(w)}
def group(a,b): return ('stable_correct' if a else 'stable_wrong') if a==b else ('rescue' if b else 'damage')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--results',required=True);ap.add_argument('--labels',required=True);ap.add_argument('--cohort',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    root=Path(a.results);labels={(x['split'],x['id']):x for x in map(json.loads,Path(a.labels).read_text(encoding='utf-8-sig').splitlines())}
    cohort=read(a.cohort);expected=set(cohort['ids']);seen=set();fail=[];buckets=collections.defaultdict(list);examples=[];drift=[];manifest=[];precision=[];controlled=collections.defaultdict(list);paired=[]
    def check(ok,msg):
        if not ok:fail.append(msg)
    def verify_scores(row,tag):
        v=row['candidate_logits'];s=row['scores'];check(len(v)==len(s) and all(math.isfinite(x) for x in v+s),tag+' shape/finite')
        check(max(abs((s[i]-s[0])-(v[i]-v[0])) for i in range(len(v)))<2e-5,tag+' FP32 score gaps')
        check(argmax(v)==argmax(s),tag+' argmax')
        check([i for i,x in enumerate(v) if x==max(v)]==[i for i,x in enumerate(s) if x==max(s)],tag+' ties')
        if 'logsumexp_fp32' in row:check(max(abs(v[i]-row['logsumexp_fp32']-s[i]) for i in range(len(v)))<2e-5,tag+' normalization')
    for p in sorted(root.glob('*.results.json')):
        obj=read(p);ident=obj['id'];tag=obj['split']+'/'+ident;check(ident not in seen,tag+' duplicate');seen.add(ident)
        lab=labels[(obj['split'],ident)];gold=lab['gold'];np=p.with_name(p.name.replace('.results.json','.native.json'));nat=read(np)
        check(sha(np)==obj['native_file_sha256'],tag+' native hash');check(nat['identity']==obj['identity'],tag+' identity')
        native=nat['native'];identityrows=[r for r in obj['rows'] if r['condition']=='identity'];check(sorted(r['receiver_loop'] for r in identityrows)==[1,2,3,4],tag+' identity coverage')
        for r in identityrows:check(r['full_logits_sha256']==native[str(r['receiver_loop'])]['full_logits_sha256'],tag+' identity logits');check(r['outside_R_unchanged'],tag+' identity non-R')
        for k,r in native.items():verify_scores(r,tag+' native '+k)
        pred=[argmax(native[str(i)]['scores']) for i in range(1,5)];old=lab['historical_predictions']
        if pred!=old:drift.append({'id':ident,'split':obj['split'],'old':old,'new':pred})
        required={(i,j,c) for i,j in [(1,2),(2,1),(2,3),(3,2),(3,4),(4,3)] for c in ['target','norm_only','unrelated_delta']}
        actual=[(r['receiver_loop'],r['donor_loop'],r['condition']) for r in obj['rows'] if r['condition']!='identity'];check(len(actual)==len(set(actual)),tag+' duplicate condition');check(required<=set(actual),tag+' condition coverage')
        lookup={(r['receiver_loop'],r['donor_loop'],r['condition']):r for r in obj['rows']}
        for r in obj['rows']:
            if r['condition']=='identity':continue
            i,j=r['receiver_loop'],r['donor_loop'];c=r['condition'];base=native[str(i)];verify_scores(r,tag+' '+str((i,j,c)))
            check(r['native']['full_logits_sha256']==base['full_logits_sha256'],tag+' recipient baseline');check(r['outside_R_unchanged'],tag+' non-R')
            check(r['direction']==('rollback' if i>j else 'restore'),tag+' direction');check(r['pair']==f'{min(i,j)}-{max(i,j)}',tag+' pair')
            check((r['donor_id']!=ident) if c=='unrelated_delta' else (r['donor_id']==ident),tag+' donor identity')
            fixed=max((k for k in range(len(base['scores'])) if k!=gold),key=lambda k:base['scores'][k]);global_fixed=max((k for k in range(len(base['scores'])) if k!=gold),key=lambda k:native['1']['scores'][k])
            bm=metrics(base['scores'],gold,fixed);im=metrics(r['scores'],gold,fixed);raw=metrics(r['candidate_logits'],gold,fixed);rb=metrics(base['candidate_logits'],gold,fixed)
            g=group(pred[min(i,j)-1]==gold,pred[max(i,j)-1]==gold);g12=group(pred[0]==gold,pred[1]==gold)
            rec={'id':ident,'split':obj['split'],'pair':r['pair'],'direction':r['direction'],'condition':c,'group':g,'group12':g12,
                 'correct':im['correct'],'rescue':int(not bm['correct'] and im['correct']),'damage':int(bm['correct'] and not im['correct']),'ties':im['ties'],
                 'global_fixedwrong_delta':metrics(r['scores'],gold,global_fixed)['fixedwrong']-metrics(base['scores'],gold,global_fixed)['fixedwrong'],
                 'wall_seconds':r['wall_seconds']}
            exact=metrics([Fraction(x) for x in r['candidate_logits']],gold,fixed);exactbase=metrics([Fraction(x) for x in base['candidate_logits']],gold,fixed)
            for metric in ['meanwrong','fixedwrong','dynamicmargin']:
                rec[metric]=im[metric];rec[metric+'_delta']=im[metric]-bm[metric];rec[metric+'_raw_delta']=raw[metric]-rb[metric]
                ed=exact[metric]-exactbase[metric];rec[metric+'_raw_sign']=(ed>0)-(ed<0)
            if 'scores_bf16_sensitivity' in r:
                rec['bf16_scoring_prediction_changed']=int(argmax(r['scores_bf16_sensitivity'])!=argmax(r['scores']))
                rec['bf16_scoring_ties_changed']=int(metrics(r['scores_bf16_sensitivity'],gold,fixed)['ties']!=im['ties'])
            for strat in ['all','adjacent:'+g,'k1k2:'+g12]:buckets[(obj['split'],r['pair'],r['direction'],c,strat)].append(rec)
            why=[]
            if c=='target':
                if r['direction']=='rollback' and rec['meanwrong_delta']>0:why.append('rollback_improves_meanwrong')
                if r['direction']=='restore' and rec['damage']:why.append('restore_damages')
                if rec['meanwrong_delta']>0 and rec['dynamicmargin_delta']<0:why.append('meanwrong_dynamic_disagree')
                other=lookup[(i,j,'unrelated_delta')];om=metrics(other['scores'],gold,fixed)
                if r['direction']=='restore' and im['meanwrong']<=om['meanwrong']:why.append('restore_not_more_beneficial_than_unrelated_meanwrong')
                if r['direction']=='rollback' and im['meanwrong']>=om['meanwrong']:why.append('rollback_not_more_harmful_than_unrelated_meanwrong')
                for control in ['norm_only','unrelated_delta']:
                    cm=metrics(lookup[(i,j,control)]['scores'],gold,fixed)
                    controlled[(obj['split'],r['pair'],r['direction'],control)].append({m:im[m]-cm[m] for m in ['meanwrong','fixedwrong','dynamicmargin']})
            if why:examples.append({**rec,'reasons':why,'native_scores':base['scores'],'intervened_scores':r['scores']})
            if c=='fp32_sdpa_bf16_writeback':
                target=lookup[(i,j,'target')];precision.append({'id':ident,'pair':r['pair'],'direction':r['direction'],'prediction_changed':argmax(r['scores'])!=argmax(target['scores']),'max_candidate_logit_change':max(abs(x-y) for x,y in zip(r['candidate_logits'],target['candidate_logits']))})
        for lo in (1,2,3):
            hi=lo+1;forward=lookup[(lo,hi,'target')];backward=lookup[(hi,lo,'target')]
            paired.append({'id':ident,'split':obj['split'],'pair':f'{lo}-{hi}','native_group':group(pred[lo-1]==gold,pred[hi-1]==gold),
                           'restore_correct':argmax(forward['scores'])==gold,'rollback_correct':argmax(backward['scores'])==gold})
        manifest.append({'id':ident,'split':obj['split'],'sha256':sha(p),'rows':len(obj['rows'])})
    summary=[]
    for k,rows in sorted(buckets.items()):
        s=dict(zip(['split','pair','direction','condition','stratum'],k));s['n']=len(rows)
        for n in ['correct','rescue','damage','ties']:s[n]=sum(r[n] for r in rows)
        s['net']=s['rescue']-s['damage'];s['wall_seconds']=sum(r['wall_seconds'] for r in rows)
        for m in ['meanwrong_delta','fixedwrong_delta','dynamicmargin_delta','global_fixedwrong_delta']:
            v=[r[m] for r in rows];s[m]={'mean':statistics.mean(v),'median':statistics.median(v),'positive':sum(x>0 for x in v),'negative':sum(x<0 for x in v),'zero':sum(x==0 for x in v)}
        for m in ['meanwrong','fixedwrong','dynamicmargin']:
            v=[r[m+'_raw_sign'] for r in rows];s[m+'_exact_raw_signs']={'positive':v.count(1),'negative':v.count(-1),'zero':v.count(0),'raw_zero_score_nonzero':sum(r[m+'_raw_sign']==0 and r[m+'_delta']!=0 for r in rows)}
        for m in ['bf16_scoring_prediction_changed','bf16_scoring_ties_changed']:s[m]=sum(r.get(m,0) for r in rows)
        summary.append(s)
    check(seen<=expected,'unexpected cohort IDs')
    contrasts=[]
    for k,rows in sorted(controlled.items()):
        rec=dict(zip(['split','pair','direction','control'],k));rec['n']=len(rows)
        for m in ['meanwrong','fixedwrong','dynamicmargin']:
            vals=[r[m] for r in rows];sgn=1 if rec['direction']=='restore' else -1
            rec[m]={'target_minus_control_mean':statistics.mean(vals),'median':statistics.median(vals),'target_more_direction_specific':sum(sgn*x>0 for x in vals),'equal':sum(x==0 for x in vals),'target_less_direction_specific':sum(sgn*x<0 for x in vals)}
        contrasts.append(rec)
    result={'status':'FAILED' if fail else ('NO_RESULTS' if not seen else ('PASS_COMPLETE' if seen==expected else 'PASS_PARTIAL')),'n_expected':len(expected),'n_observed':len(seen),'missing_ids':sorted(expected-seen),'failures':fail,'summary':summary,'paired_control_contrasts':contrasts,'paired_restore_rollback':paired,'counterexamples':examples,'hardware_prediction_drift':drift,'fp32_sdpa_sensitivity':precision,'manifest':manifest}
    Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:result[k] for k in ['status','n_expected','n_observed','failures']}))
    if fail:raise SystemExit(1)
if __name__=='__main__':main()
