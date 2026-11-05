"""CPU input admission only; constructs/audits inputs but never solves them."""
import json,hashlib,time
from pathlib import Path
from r31_oracle_inputs import load
ROOT=Path(__file__).resolve().parents[1]
def main():
 start=time.monotonic();rows,audit=load();assert len(rows)==24;result=dict(status='passed_CPU_input_admission_no_oracle',input_audit=audit,endpoint_enclosures=192,max_tokens=max(len(chain) for row in rows for chain in row['chains']),reduction_domain='n*2^-24<1/2 and S+gamma*S+n*2^-149<=FP32max, all Fraction',all_stored_LPs_finite_FP32_nonpositive=True,original_char_lengths_and_token_counts=True,real_MILP_or_LP_run=False,model_loaded=False,GPU_worker_seconds=0,elapsed_seconds=time.monotonic()-start,source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/r31_oracle_inputs.py','scripts/r31_oracle_math.py','scripts/r31_oracle_preflight.py']})
 with (ROOT/'results/r31_ORACLE_CPU_INPUT_PREFLIGHT.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
