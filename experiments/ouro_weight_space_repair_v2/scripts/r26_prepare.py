"""Prepare reviewed definition only; does not load model or run forward."""
import json
from r26_common import ROOT,sha,read,save

def main():
 old=read('configs/r25_HELLINGER_MEMORY_LOCK.json');ledgers=dict(old['budget']['inherited_ledger_sha256']);ledgers['logs/r25_gpu_budget.json']=sha('logs/r25_gpu_budget.json');assert len(ledgers)==14
 total=sum(read(p)['used_seconds'] for p in ledgers);assert total==18819.051603254396
 sources={p.relative_to(ROOT).as_posix():sha(p.relative_to(ROOT).as_posix()) for p in sorted((ROOT/'scripts').glob('r26_*.py'))}
 inherited=['scripts/runtime.py','scripts/r7_core.py','scripts/ouro_cached_scoring.py','scripts/weight_edits.py']
 inputs=['r26_CAUSAL_ACCEPTANCE_DESIGN.md','results/r26_ROOT_OFFICIAL_SOURCE_AUDIT.json','results/r26_CPU_CHECKS.json','results/r26_SANITIZED_SOURCE_AUDIT.json','results/r26_CPU_INPUT_PREFLIGHT.json','configs/r25_HELLINGER_MEMORY_LOCK.json','configs/r25_MEMORY_MANIFEST.json','results/r25_INPUT_AUDIT.json','configs/r24_INPUT_MANIFEST.json','results/slcd_SOURCE_AUDIT.json','configs/r7_reference_manifest.json','data/slcd_discovery.jsonl','data/slcd_selection.jsonl']+inherited
 for f in old['model_identity']['metadata_and_tokenizer_sha256']:
  inputs.append('models/Ouro-1.4B/'+f)
 local={p:sha(p) for p in inputs if (ROOT/p).exists()}
 for f,h in old['model_identity']['metadata_and_tokenizer_sha256'].items():local['models/Ouro-1.4B/'+f]=h
 spec=dict(status='DRAFT_PENDING_ROOT_ASTRA_REVIEW',cohort=old['cohort'],source_sha256=sources,input_sha256=local,model_identity=old['model_identity'],runtime_identity=dict(model_source_sha256='c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413',sdpa_sha256='d16bbf0bc9fdd197cf02b2917959d2d2c17b8c95555432aa27b4cde83a140033',masking_sha256='e29afe5c7058506b22c0e35e5efd8bb749dcfc2e79f32643d442b3e59088eb70',transformers='4.55.4',torch='2.11.0+cu130'),budget=dict(inherited_ledger_sha256=ledgers,inherited_total_seconds=total,remaining_seconds=21600-total,hard_seconds=120,soft_seconds=105,kill_seconds=119,runtime_reserve_seconds=5,reap_reserve_seconds=1),protocol='Five contexts Q5,N0124,key4; 6 original forwards each; 126 protected comparisons; repeated exact prerequisite; no tolerance/labels/predictions/accuracy; no retry',heldout_access_authorized=False,external_benchmarks_authorized=False,private_sources_publish=False)
 save('configs/r26_REVIEW_DRAFT.json',spec);print('DRAFT',sha('configs/r26_REVIEW_DRAFT.json'))
if __name__=='__main__':main()
