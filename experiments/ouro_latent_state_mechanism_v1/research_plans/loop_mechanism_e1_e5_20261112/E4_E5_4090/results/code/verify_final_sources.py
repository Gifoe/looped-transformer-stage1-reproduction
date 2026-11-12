"""Post-run immutable checkpoint, source and runtime-identity audit; no model is loaded."""
import json,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent;NS=ROOT/'repo/experiments/ouro_latent_state_mechanism_v1/research_plans/loop_mechanism_e1_e5_20261112'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
model={n:{'bytes':(ROOT/'model'/n).stat().st_size,'sha256':sha(ROOT/'model'/n)} for n in ('model.safetensors','modeling_ouro.py','configuration_ouro.py','config.json','tokenizer.json','tokenizer_config.json','merges.txt','vocab.json','special_tokens_map.json')}
assert model['model.safetensors']['sha256']=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
fixed={'runner_sha256':'runner.py','base_runner_sha256':'runner.py','observer_sha256':'observer.py','attestation_wrapper_sha256':'attested_runner.py','q_components_sha256':'q_patch_components.py','mediation_components_sha256':'state_mediation_components.py','base_confirmation_sha256':'confirmation_base_runner.py'};checked=[]
for path in sorted((NS/'results').glob('*/receipt.json')):
 m=json.loads(path.read_text());identity=m['identity'];assert m['status']=='COMPLETE';assert identity['model_sha256']==model['model.safetensors']['sha256'] and identity['model_source_sha256']==model['modeling_ouro.py']['sha256'] and identity['config_sha256']==model['config.json']['sha256'];pairs=fixed.copy();name=path.parent.name
 if 'query_runner_sha256' in identity:pairs['query_runner_sha256']='confirmation_query_factorial_runner.py' if name.startswith('confirmation_') else 'query_factorial_runner.py'
 if 'mediation_runner_sha256' in identity:pairs['mediation_runner_sha256']='confirmation_state_mediation_runner.py' if name.startswith('confirmation_') else ('depth_state_mediation_runner.py' if name.startswith('depth') else 'state_mediation_runner.py')
 for key,file in pairs.items():
  if key in identity:assert identity[key]==sha(ROOT/file),(name,key,file)
 checked.append({'run':name,'receipt_sha256':sha(path),'runtime_source_fields_verified':[k for k in pairs if k in identity]})
assert len(checked)==23
out={'status':'ALL_23_RUN_IDENTITIES_AND_ORIGINAL_CHECKPOINT_UNCHANGED','scope':'FILE_AND_RECORDED_RUNTIME_PROVENANCE_ONLY_NO_MODEL_FORWARD','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'fixed_revision_label':'574fa66cb8bf5abdc979642d01cf2b79b16bfab1','model_files':model,'runs':checked}
(NS/'results/provenance/FINAL_MODEL_AND_SOURCE_AUDIT.json').write_text(json.dumps(out,indent=2));print(out['status'])
