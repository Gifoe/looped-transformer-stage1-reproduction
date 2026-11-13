"""Exact no-refit full-answer replay using released low-dimensional data only."""
import argparse,json
from pathlib import Path
import numpy as np
from E6B_TRAINING_FREE_CONTROLLER import predict
from E6B_ANALYSIS import diagnostic,apply,macro,sha
from E6B_REPLAY_RELEASE import lines
def main():
 p=argparse.ArgumentParser();p.add_argument('directory');a=p.parse_args();d=Path(a.directory);f=json.loads((d/'E6B_FROZEN_SPEC.json').read_text());transfer=json.loads((d/'E6B_FAS_TRANSFER_FREEZE.json').read_text());assert sha(d/'E6B_FROZEN_SPEC.json')==transfer['hashes']['joint_freeze_sha256'];assert sha(d/'SUPERVISED_DIAGNOSTIC_MODELS.npz')==transfer['hashes']['joint_models_sha256'];labels={(r['split'],r['id']):r['gold'] for r in map(json.loads,lines(d/'E6B_LABELS.jsonl'))};features=[r for r in map(json.loads,lines(d/'E6B_FAS_FEATURES.jsonl')) if r['split']=='test'];saved={(r['id'],r['method_index']):r for r in map(json.loads,lines(d/'E6B_FAS_CONTROLLER_RESULTS.jsonl')) if r['split']=='test'};assert len(features)==1172;n=0
 for r in features:
  for i,s in enumerate(f['zero_label_specs']+f['selected_cal_specs']):
   v=predict(r['native_scores'],r['latent'],f['thresholds_unlabeled_train'][str(s['q'])],s['strategy'],s['use_latent'],s['p']);ref=saved[(r['id'],i)];assert ref['score_protocol']=='full_answer_character_mean'
   for k in ['prediction','selected_depth','executed_loops']:assert v[k]==ref[k],(r['id'],i,k)
   n+=1
 rows=[{'id':r['id'],'gold':labels[('test',r['id'])],'pred':r['native_predictions'],'scores':r['native_scores'],'latent':r['latent'],'unrelated':r['unrelated']} for r in features];coef=np.load(d/'SUPERVISED_DIAGNOSTIC_MODELS.npz');expected=json.loads((d/'E6B_FAS_DIAGNOSTICS_TRANSFER.json').read_text())
 for i,s in enumerate(f['diagnostics']):
  x,y,_=diagnostic(rows,s['setting'],s['family'],np.random.default_rng(33).permutation(len(rows)));yp=apply(tuple(coef[str(i)+'_'+k] for k in ['mean','scale','coef']),x);assert abs(macro(y,yp)-expected[i]['test_macro_f1'])<1e-12
 receipt={'status':'PASS','controller_decisions_exact':n,'supervised_diagnostics_exact':len(expected),'new_fits':0,'gpu_required':False,'source_sha256':sha(__file__),'joint_freeze_sha256':sha(d/'E6B_FROZEN_SPEC.json'),'FAS_transfer_freeze_sha256':sha(d/'E6B_FAS_TRANSFER_FREEZE.json')};(d/'E6B_FAS_RELEASE_REPLAY_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
