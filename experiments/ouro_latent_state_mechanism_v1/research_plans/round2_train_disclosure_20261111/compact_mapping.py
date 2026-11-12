"""Exact row projection used for train_scores_compact.jsonl.gz; no val/test."""
def compact(j):
 r=j['result']
 return {'id':j['id'],'is_demonstration':j['is_demonstration'],'metadata_sha256':j['metadata_sha256'],'logits_sha256':j['logits_sha256'],'gold':r['gold'],'fixed_wrong':r['k1_highest_wrong'],'baseline_K1':r['baseline_K1'],'baseline_K2':r['baseline_K2'],'conditions':r['conditions'],'norm_conditions':r['norm_conditions'],'norm_max_abs_errors':{k:v['max_abs_actual_minus_reference'] for k,v in r['norms'].items()}}
