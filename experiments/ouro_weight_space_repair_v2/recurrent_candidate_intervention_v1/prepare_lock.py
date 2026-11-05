"""Freeze-ready CPU packet compiler; does not authorize or start model work."""
from pathlib import Path
import json,hashlib,ast
ROOT=Path(__file__).resolve().parent.parent;OUT=Path(__file__).resolve().parent
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def main():
 sources=['construct_semantics.py','build_protocol.py','core.py','worker.py','supervise.py','analyze.py','cpu_checks.py','prepare_lock.py']
 for n in sources:ast.parse((OUT/n).read_text(encoding='utf8'))
 cpu=read(OUT/'CPU_CHECKS.json');assert cpu['status']=='passed_CPU_synthetic_no_model_or_GPU'
 for n,h in cpu['source_sha256'].items():assert H(OUT/n)==h
 resource=read(OUT/'PARENT_RESOURCE_VERIFICATION.json');old=resource['old_ledger_sha256'];assert len(old)==20 and sum(read(ROOT/n)['used_seconds'] for n in old)==20889.326696651056
 for n,h in old.items():assert H(ROOT/n)==h
 newinputs=['COHORT_PREREGISTRATION.json','SEMANTIC_CONSTRUCTOR_REVIEW.json','ASTRA_SEMANTIC_REVIEW.json','ASTRA_DESIGN_REVIEW.md','ASTRA_SOURCE_REVIEW.md','INFERENCE_TASK_PLAN.json','INFERENCE_REQUEST_TEXTS.json','CONSTRUCTOR_WITH_LABELS.json','prompt_and_target_audit.json','intervention_protocol.json','STATISTICAL_DEFINITIONS.md','NUMERICAL_CONTRACT.md','ROOT_A_EVIDENCE_VERIFICATION.json','ROOT_SHA_COHORT.json','ROOT_INPUT_GRAPH_REVIEW.json','ROOT_64_COST_PLAN.json']
 oldinputs=['scripts/runtime.py','scripts/ouro_cached_scoring.py','scripts/weight_edits.py','scripts/r7_inference.py','scripts/r10_core.py','scripts/r7_core.py','configs/r10_PROTOCOL.json','configs/r11_PROTOCOL.json','data/slcd_fixed_demonstrations.json','data/slcd_discovery.jsonl','data/slcd_selection.jsonl','recurrent_decision_audit_v1/collect_a4.py','recurrent_decision_audit_v1/a4_COMPLETE.json','recurrent_decision_audit_v1/a4_TIMINGS.json','recurrent_decision_audit_v1/recovered/results/r10_development_initial.jsonl','recurrent_decision_audit_v1/recovered/results/r10_development_initial_COMPLETE.json','recurrent_decision_audit_v1/inherited_recovery_receipt.json']
 inputs={n:H(ROOT/n) for n in oldinputs};inputs.update({'recurrent_candidate_intervention_v1/'+n:H(OUT/n) for n in newinputs})
 identity=read(ROOT/'configs/r11_PROTOCOL.json')['model_identity'];inputs.update({'models/Ouro-1.4B/'+n:h for n,h in identity['metadata_and_tokenizer_sha256'].items()})
 proto=read(OUT/'intervention_protocol.json');assert proto['summary']['unique_body_forwards']==218 and proto['cost']['forecast_total_seconds']==523.9555901013214 and proto['cost']['forecast_total_seconds']+5<665
 d=dict(status='review_draft_no_START',source_sha256={n:H(OUT/n) for n in sources},input_sha256=inputs,inherited_ledgers=old,inherited_used_seconds=20889.326696651056,remaining_seconds=710.6733033489436,budget=dict(hard_seconds=680,soft_TERM_seconds=665,KILL_seconds=679,reap_seconds=1,stop_new_before_soft_seconds=5,nice=15,original_cap_seconds=21600),model_identity=identity,CPU_receipt_sha256=H(OUT/'CPU_CHECKS.json'),plan_sha256=H(OUT/'INFERENCE_TASK_PLAN.json'),scientific_scope='One fixed SHA24 pilot, complete main7conditions/secondaryQ,Nfirst4/B5full2x2first4/B6lookup+copyfirst4; no decoder/calibration/training; nativeK1..4 original depth4 path',pre_effect_counts=proto['summary'],cost_policy=proto['cost'],stop='All24P0 native1/4/Fixed strict first; runtime admission same1.5warm rule then all194 remaining or no effects. Any incomplete/numerical/parity/source/parameter/timeout failure no partial results/no retry. No external benchmark/other dataset/full64/modeljob.',statistics_sha256=H(OUT/'STATISTICAL_DEFINITIONS.md'),numerical_contract_sha256=H(OUT/'NUMERICAL_CONTRACT.md'))
 (OUT/'EXECUTION_DRAFT.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(draft_sha256=H(OUT/'EXECUTION_DRAFT.json'),source_count=len(sources),input_count=len(inputs),CPU_sha256=H(OUT/'CPU_CHECKS.json'),plan_sha256=d['plan_sha256'])))
if __name__=='__main__':main()
