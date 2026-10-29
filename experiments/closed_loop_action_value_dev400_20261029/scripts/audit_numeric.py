"""Isolate one observed numerical parity failure without consulting gold."""
from run_development import *
import score
def main():
    begin=time.monotonic()
    with (ROOT/'results/development.jsonl').open(encoding='utf8') as f:done=[json.loads(l) for l in f]
    with (OLD/'data/discovery.jsonl').open(encoding='utf8') as f:rows=[json.loads(l) for l in f]
    with (OLD/'results/discovery.jsonl').open(encoding='utf8') as f:refs={r['id']:r['views']['official']['sum_scores']['adaptive'] for r in map(json.loads,f)}
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False
    directory=ROOT.parent/'looped_stage1_trajectory_audit_seed2021_v1/models/Ouro-1.4B'
    tok=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(directory,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval();model.requires_grad_(False)
    report=[]
    for row in [rows[len(done)-1],rows[len(done)]]:
        row=dict(row);row.pop('gold');pairs=row['variants']['official']
        dev,_,_,_,_=evaluate(tok,model,pairs,0.)
        prior=score.score_view(tok,model,pairs,row['question'])
        pil,_,_=frozen.evaluate(tok,model,pairs,0.)
        p,t=frozen.encode(tok,*pairs[0]);x=torch.tensor([p+t[:-1]],device='cuda')
        _,native,_=model.model(input_ids=x,use_cache=False)
        hook=frozen.Intervention(0.);handle=model.model.norm.register_forward_hook(hook)
        try:_,zero,_=model.model(input_ids=x,use_cache=False)
        finally:handle.remove()
        ref=refs[row['id']];counts=prior['token_counts']
        record=dict(id=row['id'],candidate_tokens=counts,sealed_scores=ref,current_original_scorer=prior['sum_scores']['adaptive'],development_scorer=dev,pilot_scorer=pil,zero_hook_states_exact=all(torch.equal(a,b) for a,b in zip(native,zero)),max_abs_error=max(abs(a-b) for a,b in zip(dev,ref)),max_per_token_error=max(abs(a-b)/n for a,b,n in zip(dev,ref,counts)),predictions_agree=len({int(np.argmax(np.array(v)/row['choice_char_lengths'])) for v in [ref,dev,pil,prior['sum_scores']['adaptive']]})==1)
        report.append(record)
    save('NUMERIC_PARITY_AUDIT.json',dict(completed_before_failure=len(done),partial_sha256=sha(ROOT/'results/development.jsonl'),rows=report,gold_accessed=False,elapsed_s=time.monotonic()-begin))
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
