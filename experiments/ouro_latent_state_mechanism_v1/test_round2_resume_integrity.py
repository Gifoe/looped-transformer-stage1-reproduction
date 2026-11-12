"""Fail-closed resume schema checks; synthetic only."""
import tempfile,json,hashlib
import torch
from safetensors.torch import save_file
from pathlib import Path
from round2_intervention_runner import validate_envelope,canonical
from round2_intervention_core import CORE,NORM
identity={'id':'case'};records=[{'candidate':i,'encoding':{'input_ids':[1,2],'predecessor_positions':[1],'target_ids':[i+1]},'scores':[float(-i),float(-i)]} for i in range(2)];r={'gold':0,'target_ids':[1,2],'conditions':{k:{} for k in CORE},'norm_conditions':{k:{} for k in NORM},'controls':{},'input_length':2,'readout_positions':[1],'context_positions_count':1,'baseline_K1':{'scores':[0.0,-1.0]},'baseline_K2':{'scores':[0.0,-1.0]}};payload={'identity':identity,'result':r}
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'case.json';p.write_text(json.dumps({'payload':payload,'payload_sha256':hashlib.sha256(canonical(payload)).hexdigest()}))
 try:validate_envelope(p,identity,0,[1,2],records)
 except AssertionError:pass
 else:raise RuntimeError('empty controls were accepted')
 p.write_text(json.dumps({'payload':payload,'payload_sha256':'0'*64}))
 try:validate_envelope(p,identity,0,[1,2],records)
 except AssertionError:pass
 else:raise RuntimeError('corrupt payload hash accepted')
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'case.json';tp=p.with_suffix('.safetensors');keys={'baseline_K1','baseline_K2','negative_last_C'}|{'core_'+x for x in CORE}|{'norm_'+x for x in NORM};tensor={k:torch.zeros((1,49152),dtype=torch.bfloat16) for k in keys};save_file(tensor,str(tp))
 item={'scores':[0.0,-1.0],'prediction':0,'top_tie':False,'gold_vs_fixed_wrong_gap':1.0};norm={'source_norm':[[1.0,1.0]],'reference_norm':[[1.0,1.0]],'actual_bf16_norm':[[1.0,1.0]],'max_abs_actual_minus_reference':0.0}
 r={'gold':0,'target_ids':[1,2],'k1_highest_wrong':1,'readout_positions':[1],'context_positions':[0],'context_positions_count':1,'input_length':2,'conditions':{k:item for k in CORE},'norm_conditions':{k:item for k in NORM},'controls':{'K2_identity':'exact','K1_full_anchor':'exact','last_layer_C_only':'exact','rescale_K2_to_K2':'exact'},'norms':{'K1DIR_K2NORM':norm,'K2DIR_K1NORM':norm},'baseline_K1':item,'baseline_K2':item}
 payload={'identity':identity,'result':r,'logits_sha256':hashlib.sha256(tp.read_bytes()).hexdigest()};p.write_text(json.dumps({'payload':payload,'payload_sha256':hashlib.sha256(canonical(payload)).hexdigest()}));validate_envelope(p,identity,0,[1,2],records)
 r['baseline_K1']={**item,'scores':[0.0,-2.0],'gold_vs_fixed_wrong_gap':2.0};payload['result']=r;p.write_text(json.dumps({'payload':payload,'payload_sha256':hashlib.sha256(canonical(payload)).hexdigest()}))
 try:validate_envelope(p,identity,0,[1,2],records)
 except AssertionError:pass
 else:raise RuntimeError('changed cohort baseline accepted')
 r['baseline_K1']=item;r['readout_positions']=[0];r['context_positions']=[1];payload['result']=r;p.write_text(json.dumps({'payload':payload,'payload_sha256':hashlib.sha256(canonical(payload)).hexdigest()}))
 try:validate_envelope(p,identity,0,[1,2],records)
 except AssertionError:pass
 else:raise RuntimeError('changed readout position accepted')
 r['readout_positions']=[1];r['context_positions']=[0];payload['result']=r;p.write_text(json.dumps({'payload':payload,'payload_sha256':hashlib.sha256(canonical(payload)).hexdigest()}))
 tensor['baseline_K1']=torch.ones((1,49152),dtype=torch.bfloat16);save_file(tensor,str(tp))
 try:validate_envelope(p,identity,0,[1,2],records)
 except AssertionError:pass
 else:raise RuntimeError('changed BF16 logits accepted')
print('PASS rejects empty controls, corrupt metadata hash and changed saved BF16 logits; synthetic only')
