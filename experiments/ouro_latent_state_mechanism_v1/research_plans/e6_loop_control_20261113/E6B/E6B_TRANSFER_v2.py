"""No-refit transfer: Joint-frozen E6B rules to an independently frozen Full-Answer protocol."""
import argparse,json,time,importlib.util
from pathlib import Path
import numpy as np
from E6B_ANALYSIS_v2 import load,evaluate
from E6B_ANALYSIS import sha,write,diagnostic,apply,macro

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['freeze','evaluate'],required=True);p.add_argument('--capture',nargs='+',required=True);p.add_argument('--labels',required=True);p.add_argument('--test-inputs',required=True);p.add_argument('--joint-results',required=True);p.add_argument('--e6a-methods',required=True);p.add_argument('--e6a-spec',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);j=Path(a.joint_results);joint=json.loads((j/'E6B_FROZEN_SPEC.json').read_text());labels={(r['split'],r['id']):r for r in map(json.loads,Path(a.labels).read_text(encoding='utf-8-sig').splitlines())};splits,sources,core=load(a.capture,labels);assert core['score_protocol']=='full_answer'
 for key in ['model_sha256','gpu','dtype','attention','torch']:assert core[key]==joint['capture_identity_core'][key]
 hashes={'joint_freeze_sha256':sha(j/'E6B_FROZEN_SPEC.json'),'joint_models_sha256':sha(j/'SUPERVISED_DIAGNOSTIC_MODELS.npz'),'source_sha256':sha(__file__),'joint_evaluator_sha256':sha(Path(__file__).parent/'E6B_ANALYSIS_v2.py'),'controller_source_sha256':sha(Path(__file__).parent/'E6B_TRAINING_FREE_CONTROLLER.py'),'test_inputs_sha256':sha(a.test_inputs),'labels_sha256':sha(a.labels),'e6a_methods_sha256':sha(a.e6a_methods),'e6a_spec_sha256':sha(a.e6a_spec)}
 f=out/'E6B_FAS_TRANSFER_FREEZE.json'
 if a.phase=='freeze':
  assert [len(splits[s]) for s in ['train','validation','test']]==[1094,299,0] and not f.exists()
  write(f,{'created_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),'phase':'FAS_DEV_ONLY_FREEZE_BEFORE_NEW_TEST_CAPTURE','hashes':hashes,'capture_identity_core':core,'development_sources':sources,'rule':'Use exact previously Joint-frozen methods, thresholds and supervised coefficients. No FAS label refitting or parameter selection. Character-normalized full-answer log likelihood primary.'});print('E6B_FAS_TRANSFER_FROZEN',sha(f),flush=True);return
 freeze=json.loads(f.read_text());assert hashes==freeze['hashes'] and core==freeze['capture_identity_core'];assert not (out/'E6B_FAS_TEST_RECEIPT.json').exists();assert [len(splits[s]) for s in ['train','validation','test']]==[1094,299,1172]
 expected={(r['split'],r['id']) for r in map(json.loads,Path(a.test_inputs).read_text().splitlines())};assert {(r['split'],r['id']) for r in splits['test']}==expected
 im=importlib.util.spec_from_file_location('e6a_frozen',a.e6a_methods);mod=importlib.util.module_from_spec(im);im.loader.exec_module(mod);asp=json.loads(Path(a.e6a_spec).read_text());results,individual=evaluate(splits,joint['zero_label_specs']+joint['selected_cal_specs'],joint['thresholds_unlabeled_train'],a,asp,mod)
 for r in results:r['score_protocol']='full_answer_character_mean'
 for r in individual:r['score_protocol']='full_answer_character_mean'
 coeff=np.load(j/'SUPERVISED_DIAGNOSTIC_MODELS.npz');diag=[]
 for i,s in enumerate(joint['diagnostics']):
  model=tuple(coeff[str(i)+'_'+key] for key in ['mean','scale','coef']);rs=splits['test'];x,y,_=diagnostic(rs,s['setting'],s['family'],np.random.default_rng(33).permutation(len(rs)));pr=apply(model,x);diag.append({'setting':s['setting'],'family':s['family'],'track':'SUPERVISED_DIAGNOSTIC_FROZEN_JOINT_TRANSFER','test_accuracy':float((pr==y).mean()),'test_macro_f1':macro(y,pr)})
 write(out/'E6B_FAS_TRANSFER_SUMMARY.json',results);write(out/'E6B_FAS_DIAGNOSTICS_TRANSFER.json',diag);(out/'E6B_FAS_CONTROLLER_RESULTS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in individual),encoding='utf8');write(out/'E6B_FAS_TEST_RECEIPT.json',{'status':'COMPLETE','n':1172,'frozen_transfer_sha256':sha(f),'new_fits':0,'new_parameter_selections':0,'capture_sources':sources});print('E6B_FAS_TRANSFER_COMPLETE',len(individual),flush=True)
if __name__=='__main__':main()
