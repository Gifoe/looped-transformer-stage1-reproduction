"""Run the unchanged collector with in-process loaded-module source attestation."""
import sys,importlib
from pathlib import Path
import runner

factory=runner.AutoModelForCausalLM.from_pretrained
def attested_factory(*args,**kwargs):
 model=factory(*args,**kwargs)
 out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
 model_path=Path(importlib.import_module(model.__class__.__module__).__file__).resolve()
 config_path=Path(importlib.import_module(model.config.__class__.__module__).__file__).resolve()
 source=runner.ROOT/'model/modeling_ouro.py';configuration=runner.ROOT/'model/configuration_ouro.py'
 assert runner.sha(model_path)==runner.sha(source)
 assert runner.sha(config_path)==runner.sha(configuration)
 identity={'scope':'ACTUAL_FROM_PRETRAINED_RETURNED_OBJECT_BEFORE_CUDA_TRANSFER','model_class':model.__class__.__qualname__,'model_module':model.__class__.__module__,'loaded_model_source_path':str(model_path),'loaded_model_source_sha256':runner.sha(model_path),'loaded_config_source_path':str(config_path),'loaded_config_source_sha256':runner.sha(config_path),'on_disk_modeling_sha256':runner.sha(source),'on_disk_configuration_sha256':runner.sha(configuration),'attestation_wrapper_sha256':runner.sha(__file__)}
 p=out/'loaded_source_identity.json'
 if p.exists():assert runner.json.loads(p.read_text())==identity
 runner.write(p,identity)
 return model
runner.AutoModelForCausalLM.from_pretrained=staticmethod(attested_factory)
if __name__=='__main__':runner.main()
