"""Decode-check the exact three validation cases used by the parity gate."""
import argparse,hashlib,json,sys
from pathlib import Path
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from cohort_runner import PROJECT,REPO,split_data_path,write_json,sha
from prompt_spans import make_question,token_spans
from manifest_guard import validate_manifest
sys.path.insert(0,str(REPO/'experiments/ouro_weight_space_repair_v2/scripts'))
from r10_core import question_prompt,joint_prefix,joint_requests
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out).resolve();out.relative_to(PROJECT.resolve());here=Path(__file__).resolve().parent
 validate_manifest('validation',split_data_path('validation'),here/'results/INPUT_MANIFEST_SUMMARY.json',here/'results/INPUT_MANIFEST_validation.jsonl')
 tok=AutoTokenizer.from_pretrained(PROJECT/'model',local_files_only=True,trust_remote_code=True);prefix=joint_prefix(json.loads((REPO/'experiments/ouro_weight_space_repair_v2/data/slcd_fixed_demonstrations.json').read_text()));rows=pq.read_table(split_data_path('validation')).to_pylist()
 ordered=sorted(rows,key=lambda r:(len(tok.encode(prefix+question_prompt(r['question'],r['choices']['text']),add_special_tokens=False)),len(r['choices']['text']),r['id']));groups={n:[r for r in ordered if len(r['choices']['text'])==n] for n in (3,4,5)}
 if any(not groups[n] for n in (3,4,5)):raise RuntimeError('missing gate choice strata')
 selected=[groups[3][0],groups[4][len(groups[4])//2],groups[5][-1]];records=[]
 for row in selected:
  prompt,spans=make_question(prefix,row['question'],row['choices']['text'],question_prompt);requests=joint_requests({'question':row['question'],'choices':row['choices']['text']},prefix);role_audit=[]
  for j,request in enumerate(requests):
   if request!=(prompt,' '+chr(65+j)):raise RuntimeError('historical request mismatch')
   enc=token_spans(tok,*request,spans);offsets=enc['offset_mapping'];ids=enc['context_ids']
   for role,(start,end) in spans.items():
    positions=enc['spans'][role];decoded=tok.decode([ids[i] for i in positions],skip_special_tokens=False,clean_up_tokenization_spaces=False);wanted=prompt[start:end]
    if wanted not in decoded:raise RuntimeError(f'decoded span mismatch {row["id"]} candidate {j} role {role}')
    if any(offsets[i][0]>=end or offsets[i][1]<=start for i in positions):raise RuntimeError('offset intersection mismatch')
    role_audit.append({'candidate':j,'role':role,'characters':[start,end],'token_positions':positions,'intended':wanted,'decoded':decoded})
  records.append({'id':row['id'],'option_count':len(row['choices']['text']),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'role_audit':role_audit})
 write_json(out,{'status':'passed','scope':'three validation gate cases, decoded semantic spans; no inference','validation_data_sha256':sha(split_data_path('validation')),'records':records})
if __name__=='__main__':main()
