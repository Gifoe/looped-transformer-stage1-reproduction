"""R24 gold admission after complete576 audited label-free table only."""
import json
from r24_common import ROOT,LOCK,read,sha,save,gate

def main():
 spec=gate();d=read('results/r24_SUPPORT_COMPLETE.json');u=read('logs/r24_SUPERVISOR.json');assert d['status']=='complete_gold_free_support' and d['n']==576 and u['status']=='complete' and not u['overrun'] and not u.get('timed_out',False)
 assert d['protocol_sha256']==sha(LOCK) and d['support_sha256']==sha('results/r24_support.jsonl') and d['value_inventory_sha256']==sha('results/r24_VALUE_INVENTORY.json')
 rows=[json.loads(x) for x in (ROOT/'results/r24_support.jsonl').read_text(encoding='utf8').splitlines()];m=read('configs/r24_INPUT_MANIFEST.json');assert [r['id'] for r in rows]==[r['id'] for r in m['rows']]
 gold={r['id']:r['gold'] for split in ['discovery','selection'] for r in [json.loads(x) for x in (ROOT/('results/r7_'+split+'_development.jsonl')).read_text(encoding='utf8').splitlines()]};inventory=read('results/r24_VALUE_INVENTORY.json');summary={}
 for split in ['discovery','selection','combined']:
  subset=[r for r in rows if split=='combined' or r['split']==split];wrong=[r for r in subset if r['R6_winner']!=gold[r['id']]];specific=sum(bool(r['candidates'][gold[r['id']]]['specific_difference_vs_R6']) for r in wrong);generic=sum(bool(r['candidates'][gold[r['id']]]['all_difference_vs_R6']) for r in wrong)
  background={}
  for group in ['all_wrong_candidates','wrong_nonanchor_candidates']:
   pairs=[(r,c) for r in subset for c in r['candidates'] if c['candidate']!=gold[r['id']] and (group=='all_wrong_candidates' or c['candidate']!=r['R6_winner'])]
   background[group]=dict(pairs=len(pairs),specific_covered_pairs=sum(bool(c['specific_difference_vs_R6']) for _,c in pairs),all_covered_pairs=sum(bool(c['all_difference_vs_R6']) for _,c in pairs),questions=len({r['id'] for r,c in pairs}),specific_covered_questions=len({r['id'] for r,c in pairs if c['specific_difference_vs_R6']}))
  summary[split]=dict(n=len(subset),R6_errors=len(wrong),specific_gold_vs_R6_covered=specific,specific_proportion=specific/len(wrong),all_gold_vs_R6_covered=generic,all_proportion=generic/len(wrong),wrong_candidate_background=background)
 gates=dict(combined_specific_20percent=summary['combined']['specific_proportion']>=.2,eachsplit_specific_10percent=all(summary[s]['specific_proportion']>=.1 for s in ['discovery','selection']),specific_types_atleast10=len(inventory['S_specific'])>=10,specific_demos_atleast5=len(inventory['specific_demo_ids'])>=5,audits=True)
 save('results/r24_ADMISSION.json',dict(status='complete_CPU_support_admission',n=576,summary=summary,S_all_types=len(inventory['S_all']),S_specific_types=len(inventory['S_specific']),specific_demo_count=len(inventory['specific_demo_ids']),gates=gates,admission_passed=all(gates.values()),protocol_sha256=d['protocol_sha256'],support_sha256=d['support_sha256'],scope='Offline target-set content opportunity only; no semantic support, fullV reachability or deployment guarantee; no pvalues; no future keys/GPU authorized'))
 print('R24_ADMISSION_COMPLETE')
if __name__=='__main__':main()
