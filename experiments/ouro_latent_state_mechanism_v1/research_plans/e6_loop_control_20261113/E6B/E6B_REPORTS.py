"""Reports from completed immutable CPU evaluation; no experiment selection."""
import argparse,json,hashlib,shutil
from pathlib import Path

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def table(rows):
 out=['| Track / strategy | Latent | Correct / N | Δ K4 pp [95% CI] | Rescue / damage | Policy mean loops |','|---|---:|---:|---|---:|---:|']
 for r in rows:
  s=r['method'];ci=r['paired_question_bootstrap95_pp'];out.append(f"| {s['track']} / {s['strategy']} | {s['use_latent']} | {r['correct']}/{r['n']} | {r['pp_gain']:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}] | {r['rescue']}/{r['damage']} | {r['mean_executed_loops']:.3f} |")
 return '\n'.join(out)+'\n'
def main():
 p=argparse.ArgumentParser();p.add_argument('--joint',required=True);p.add_argument('--fas');p.add_argument('--out',required=True);a=p.parse_args();j=Path(a.joint);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);f=read(j/'E6B_FROZEN_SPEC.json');test=read(j/'E6B_TEST_SUMMARY.json');diag=read(j/'E6B_DIAGNOSTICS.json');dev=read(j/'E6B_DEVELOPMENT_SUMMARY.json');assert read(j/'E6B_TEST_RECEIPT.json')['status']=='COMPLETE'
 for name in ['E6B_PROTOCOL.md','E6B_FEATURE_DEFINITIONS.md','E6B_STATE_AVAILABILITY_AUDIT.json','E6B_NATURAL_FEATURE_EXTRACTION_v3.py','E6B_TRAINING_FREE_CONTROLLER.py','E6B_ANALYSIS.py','E6B_ANALYSIS_v2.py','E6B_TRANSFER_v2.py','E6_SHARED_FULL_ANSWER_CAPTURE_v2.py']:
  src=Path(__file__).parent/name
  if src.exists():shutil.copy2(src,out/name)
 (out/'E6B_NATURAL_FEATURE_EXTRACTION.py').write_bytes((Path(__file__).parent/'E6B_NATURAL_FEATURE_EXTRACTION_v3.py').read_bytes())
 caveat='\nThese are fixed ARC project-internal evaluations after historical ARC/Test exposure. Native/outputs/features are same RTX5090 and same score protocol. Bootstrap intervals resample questions, not training seeds; multiple methods and validation choices are disclosed. Depths are causal-prefix policy replay, not measured online latency. No original Ouro weights changed.\n'
 for name,title in [('E6B_OUTPUT_VS_LATENT_ABLATION.md','Output versus latent matched comparisons'),('E6B_ACCURACY_AND_COMPUTE.md','Accuracy and policy compute')]:
  text='# '+title+'\n\n'+table(test)+caveat+'\nValidation-calibrated gates are separate from ZL; supervised diagnostics are not controller results. LoopCD and both E6A frozen primary paired comparisons are retained in the complete machine-readable summaries. Full-budget rollback/agreement/stability execute four loops; only prefix early_exit has an expected reduced policy depth.\n';(out/name).write_text(text,encoding='utf8')
 text='# Supervised update-utility diagnostics\n\n| Timing | Features | Alpha | Validation macro-F1 | Test macro-F1 | Test accuracy |\n|---|---|---:|---:|---:|---:|\n'
 for r in diag:text+=f"| {r['setting']} | {r['family']} | {r['alpha']} | {r['validation_macro_f1']:.4f} | {r['test_macro_f1']:.4f} | {r['test_accuracy']:.4f} |\n"
 text+='\nRetrospective sees the completed update. Prospective sees only current K2/K3 features and predicts the next transition. Four classes retain stable-wrong, rescue, damage, stable-correct. Train-only standardization and supervised ridge fitting, Validation alpha choice; Test never fitted. Random sample correspondence and matched unrelated-layer controls are independent diagnostic comparators. Predictive signal alone is not zero-label controller utility.\n'+caveat;(out/'E6B_UTILITY_PREDICTION_ANALYSIS.md').write_text(text,encoding='utf8')
 counter=[]
 for r in map(json.loads,(j/'E6B_CONTROLLER_RESULTS.jsonl').read_text().splitlines()):
  if r['prediction']!=r['gold']:counter.append({k:r[k] for k in ['id','split','gold','prediction','method_index','selected_depth']})
 (out/'E6B_COUNTEREXAMPLES.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in counter),encoding='utf8');(out/'E6B_COUNTEREXAMPLES.md').write_text('# Counterexamples\n\nAll incorrect decisions for every method are retained in E6B_COUNTEREXAMPLES.jsonl. They include stable-wrong cases and regressions; never select a supported subgroup as the overall effect. Exact per-method paired damage/rescue and all no-tie summaries remain in the primary result JSON.\n',encoding='utf8')
 if a.fas:
  fs=read(Path(a.fas)/'E6B_FAS_TRANSFER_SUMMARY.json');(out/'E6B_FULL_ANSWER_TRANSFER.md').write_text('# Frozen transfer to full-answer scoring\n\n'+table([r for r in fs if r['split']=='test'])+'\nComplete answer-text full-vocabulary conditional log likelihood, normalized by answer character count. Same25 demos use full-answer completions. This differs from Joint-option letter scoring; no values are mixed. Joint-frozen rules, thresholds and supervised coefficients transfer with zero refits or FAS parameter changes. The observable R vectors come from fixed candidate0 causal prefix; shape-dependent numeric differences are possible and attested by separate technical native tests.\n'+caveat,encoding='utf8')
 else:(out/'E6B_FULL_ANSWER_TRANSFER.md').write_text('# Full-answer transfer\n\nNOT RUN at this report snapshot; no Joint result substitutes for the independent full-answer protocol.\n',encoding='utf8')
 primary=next(r for r in test if r['method']['track']=='ZL' and r['method']['strategy']=='rollback' and r['method']['use_latent'])
 strongest=[primary]+list(primary['relative_frozen_E6A'].values());nominal=all(r['net_gain']>0 for r in strongest);robust=nominal and all(r['paired_question_bootstrap95_pp'][0]>0 for r in strongest)
 decision='HOLD_FOR_INDEPENDENT_CONFIRMATION' if robust else 'NO_GO_CURRENT_EVIDENCE'
 (out/'E6C_GO_NO_GO_DECISION.md').write_text('# E6-C decision\n\n'+decision+'\n\nThe preregistered ZL rollback+latent primary has '+str(primary['net_gain'])+' net correct versus same-device K4. Nominal gains versus every frozen E6A comparator: '+str(nominal)+'. All paired95% lower bounds above zero: '+str(robust)+'. Supervised diagnostic success is not counted as zero-label control success. Other calibrated/exploratory methods remain visible; no hidden-state intervention or external benchmark is automatically launched.\n',encoding='utf8')
 (out/'E6B_ASTRA_SCIENTIFIC_REVIEW.md').write_text('# Current-model scientific self-review (not Astra)\n\nAll E6 analysis, implementation, execution and this self-review were performed by Sol. Astra was subsequently authorized separately for next-step direction only, without E6 analysis/code/GPU work.\n\n'+(out/'E6C_GO_NO_GO_DECISION.md').read_text()+caveat,encoding='utf8')
 for name in ['E6B_FROZEN_SPEC.json','E6B_TEST_SUMMARY.json','E6B_DIAGNOSTICS.json','E6B_DEVELOPMENT_SUMMARY.json','E6B_CONTROLLER_RESULTS.jsonl','E6B_TEST_RECEIPT.json','SUPERVISED_DIAGNOSTIC_MODELS.npz']:shutil.copy2(j/name,out/name)
 print('E6B_REPORTS_COMPLETE',decision)
if __name__=='__main__':main()
