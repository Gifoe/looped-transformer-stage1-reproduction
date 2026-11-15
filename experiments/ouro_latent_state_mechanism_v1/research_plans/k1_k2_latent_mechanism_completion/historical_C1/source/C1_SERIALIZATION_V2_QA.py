"""CPU scalar/vector byte and lossless archive regression test; never loads a model."""
from pathlib import Path
import json,hashlib
import torch
import branch_interfaces as old
import branch_interfaces_v2 as new
import C1_COLLECT_v2 as collect
r=Path(__file__).parent;qa=r/'serialization_qa';qa.mkdir(exist_ok=True);assert not torch.cuda.is_initialized()
t={'scale_fp64_0d':torch.tensor(1.2345678901234567,dtype=torch.float64),'scalar_BF16_0d':torch.tensor(-0.,dtype=torch.bfloat16),'vector_BF16':torch.tensor([0.,-0.,1.,-2.5,3.75],dtype=torch.bfloat16),'matrix_FP64':torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float64)}
meta=collect.save(qa/'v2.safetensors',t,r);back=collect.load_tensor(qa/'v2.safetensors')
for key,v in t.items():assert new.bits(v,back[key]) and new.thash(v)==new.thash(back[key])
for key in ['vector_BF16','matrix_FP64']:assert old.thash(t[key])==new.thash(t[key]) and old.bits(t[key],back[key])==new.bits(t[key],back[key])
assert meta['tensor_sha256']==collect.tensor_sha(qa/'v2.safetensors');assert meta['tensor_archive_sha256']==collect.sha(r/meta['tensor_file']);assert not torch.cuda.is_initialized()
receipt={'status':'PASS','scope':'CPU only, synthetic0D FP64/0D BF16/vectorBF16/matrixFP64; no model or new scientific outcomes','all_original_shapes_and_bits_roundtrip':True,'non_scalar_payload_hashes_unchanged_from_v1':True,'CUDA_initialized':False,'new_GPU_forwards':0,'collector_v2_sha256':collect.sha(r/'C1_COLLECT_v2.py'),'helper_v2_sha256':collect.sha(r/'branch_interfaces_v2.py'),'archive':meta};(qa/'V2_CPU_SERIALIZATION_QA.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
