"""CPU-only protocol and data audit; does not assess model or GPU parity."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,sys
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from lm_eval.api.model import TemplateLM
ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT.parent
sys.path.insert(0,str(ROOT/'experiments/ouro_weight_space_repair_v2/scripts'))
from r10_core import joint_prefix,joint_requests
demo_path=ROOT/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json'
demos=json.loads(demo_path.read_text(encoding='utf-8'))
prefix=joint_prefix(demos)
rows=pq.read_table(PROJECT/'data/validation.parquet').to_pylist()
row=rows[0]; clean={'question':row['question'],'choices':row['choices']['text']}
requests=joint_requests(clean,prefix)
tokenizer=AutoTokenizer.from_pretrained(PROJECT/'model',local_files_only=True,trust_remote_code=True)
coder=SimpleNamespace(tok_encode=lambda text,**kw:tokenizer.encode(text,add_special_tokens=False))
pairs=[TemplateLM._encode_pair(coder,*pair) for pair in requests]
assert len(demos['ids'])==25 and len(set(demos['ids']))==25
assert len(requests)==len(clean['choices']) and all(p and t for p,t in pairs)
record={'scope':'CPU protocol smoke test only; not GPU parity or benchmark','demo_sha256':hashlib.sha256(demo_path.read_bytes()).hexdigest(),'demo_ids':demos['ids'],'prefix_sha256':hashlib.sha256(prefix.encode()).hexdigest(),'sample_id':row['id'],'requests_sha256':[hashlib.sha256(json.dumps(x,ensure_ascii=False).encode()).hexdigest() for x in requests],'continuation_token_ids':[x[1] for x in pairs],'input_lengths':[len(x[0])+len(x[1]) for x in pairs],'bos_token_id':tokenizer.bos_token_id,'eos_token_id':tokenizer.eos_token_id}
out=Path(__file__).parent/'results/CPU_PROTOCOL_AUDIT.json';out.write_text(json.dumps(record,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');print(json.dumps(record,ensure_ascii=False))
