"""Fixed original E2 heads: descriptive candidate-conditional metrics, never refits."""
import pathlib,sys,json,hashlib,math,statistics,os
from collections import defaultdict
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--stage',required=True);args=ap.parse_args();B=pathlib.Path(args.root);S=args.stage;D=B/'analysis_results'/S;Z=B/'data'/S;sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
 source=D/'SOURCE.json';rowsfile=D/'rows.jsonl';audit=Z/'INDEPENDENT_COVERAGE_AUDIT.json';p=Z/'PENDING.json'; start={str(x):sha(x) for x in [source,rowsfile,audit,p,pathlib.Path(__file__)]};s=json.loads(source.read_text());assert s['rows_sha256']==sha(rowsfile) and s['pending_sha256']==sha(p) and s['audit_sha256']==sha(audit);rows=[json.loads(x) for x in rowsfile.read_text().splitlines()]; cases=json.loads(p.read_text())['cases'];assert [r['id'] for r in rows]==[c['id'] for c in cases];records=[];primary=None
 for r,c in zip(rows,cases):
  mpath=Z/c['metadata_file'];assert sha(mpath)==c['metadata_sha256'];m=json.loads(mpath.read_text());o=r['frozen_e2'];assert o['status']=='FROZEN_E2_DESCRIPTIVE_APPLICATION' and o['secondary_all16'];primary=primary or o['primary_keys'];assert primary==o['primary_keys']
  for condition,heads in o['results'].items():
   native=m['identity']['K1_scores'] if condition=='K1' else m['conditions'][condition]['scores'];gold=r['gold']
   for head,v in heads.items():
    probs=v['probabilities'][0];scores=v['scores'][0];pred=v['prediction_offered_index'][0];ties=v['top_ties_offered_indices'][0];assert len(probs)==len(scores)==len(m['targets']) and v['offered_token_ids']==m['targets'] and abs(sum(probs)-1)<1e-8 and pred==ties[0] and 0<=gold<len(probs);conf=max(probs);assert all(0<=x<=1 for x in probs)
    records.append({'offered_token_ids':m['targets'],'offered_rank_gold':gold+1,'score_range':max(scores)-min(scores),'id':r['id'],'group':r['natural_group'],'natural_no_tie':r['natural_no_tie'],'condition':condition,'head':head,'primary':head in primary,'gold':gold,'candidate_count':len(probs),'prediction':pred,'native_prediction':native['prediction'],'probe_correct':pred==gold,'native_correct':native['prediction']==gold,'probe_tie':len(ties)>1,'native_tie':len(native['ties'])>1,'cross_entropy':-math.log(probs[gold]) if probs[gold]>0 else None,'brier':sum((v0-(1.0 if j==gold else 0.0))**2 for j,v0 in enumerate(probs)),'confidence':conf,'gold_vs_nongold_score_gap':scores[gold]-sum(x for j,x in enumerate(scores) if j!=gold)/(len(scores)-1)})
 (D/'E2_ROWS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
 summary={};buckets=defaultdict(lambda:defaultdict(list))
 for item in records:
  gs=['all',item['group']];gs += [x+'_no_tie' for x in gs] if item['natural_no_tie'] else []
  for gn in gs:buckets[gn][(item['condition'],item['head'])].append(item)
 for group in ['all','rescue','damage','stable_correct','stable_wrong']:
  for nt in [False,True]:
   gname=group+('_no_tie' if nt else '');summary[gname]={}
   for cond,head in sorted(buckets[gname]):
    rr=buckets[gname][(cond,head)]; n=len(rr);bins=[[] for _ in range(10)]
    for r in rr:bins[min(int(r['confidence']*10),9)].append(r)
    ece=sum(len(bin)/n*abs(statistics.mean(r['confidence'] for r in bin)-statistics.mean(r['probe_correct'] for r in bin)) for bin in bins if bin);ces=[r['cross_entropy'] for r in rr if r['cross_entropy'] is not None];strata={}
    for key in sorted({(r['gold'],r['candidate_count']) for r in rr}):
     v=[r for r in rr if (r['gold'],r['candidate_count'])==key];strata[str(key)]={'n':len(v),'correct':sum(r['probe_correct'] for r in v)}
    summary[gname][cond+'|'+head]={'n':n,'primary':rr[0]['primary'],'accuracy':sum(r['probe_correct'] for r in rr)/n,'native_accuracy':sum(r['native_correct'] for r in rr)/n,'native_prediction_agreement':sum(r['prediction']==r['native_prediction'] for r in rr)/n,'mean_candidate_conditional_cross_entropy':statistics.mean(ces) if ces else None,'zero_gold_probability_count':n-len(ces),'mean_candidate_conditional_brier':statistics.mean(r['brier'] for r in rr),'ece_10bin_candidate_conditional':ece,'ece_bin_count_sum':sum(len(bin) for bin in bins),'mean_gold_vs_nongold_probe_score_gap':statistics.mean(r['gold_vs_nongold_score_gap'] for r in rr),'probe_ties':sum(r['probe_tie'] for r in rr),'native_ties':sum(r['native_tie'] for r in rr),'gold_candidate_count_strata':strata,'mean_score_range':statistics.mean(r['score_range'] for r in rr),'probe_correct_native_wrong_ids':[r['id'] for r in rr if r['probe_correct'] and not r['native_correct']],'native_correct_probe_wrong_ids':[r['id'] for r in rr if r['native_correct'] and not r['probe_correct']]}
  # All condition/head rows retain observed decision transitions; no head chosen on performance.
 # Compare the same question and same fixed head before/after each boundary or Q9 intervention.
 lookup={(r['id'],r['condition'],r['head']):r for r in records};head_names=sorted({r['head'] for r in records});pairs=[('H00','H10'),('H01','H11'),('H00','H01'),('H10','H11'),('H01','U_Q9_N'),('N_Q9_U','H11')]
 for group in ['all','rescue','damage','stable_correct','stable_wrong']:
  for nt in [False,True]:
   subset=[r for r in rows if (group=='all' or r['natural_group']==group) and (not nt or r['natural_no_tie'])];gname=group+('_no_tie' if nt else '');trans={}
   for left,right in pairs:
    for head in head_names:
     pairs_ok=[(lookup[(r['id'],left,head)],lookup[(r['id'],right,head)]) for r in subset if (r['id'],left,head) in lookup and (r['id'],right,head) in lookup]; trans[left+'->'+right+'|'+head]={'n':len(pairs_ok),'probe_changed_native_unchanged_ids':sorted(a['id'] for a,b in pairs_ok if a['prediction']!=b['prediction'] and a['native_prediction']==b['native_prediction']),'native_changed_probe_unchanged_ids':sorted(a['id'] for a,b in pairs_ok if a['prediction']==b['prediction'] and a['native_prediction']!=b['native_prediction'])}
   summary[gname]['_paired_transitions']=trans
 assert all(sha(k)==v for k,v in start.items());out=D/'E2_DESCRIPTIVE.json';meta=D/'E2_DESCRIPTIVE_SOURCE.json';assert not out.exists() and not meta.exists()
 with out.open('x') as f:json.dump({'primary_head_keys':primary,'summary':summary,'scope':'Original frozen E2 heads, candidate-conditional probabilities; decode is not evidence of native causal use; unit-L2 features are followed by coordinate standardization. E-axis train had no gold positives.'},f,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 with meta.open('x') as f:json.dump({'status':'AB1_FROZEN_E2_DESCRIPTIVE_COMPLETE','source_sha256':start,'output_sha256':sha(out)},f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print('AB1_E2_DESCRIPTIVE_COMPLETE',S,sha(out),flush=True)
if __name__=='__main__':main()
