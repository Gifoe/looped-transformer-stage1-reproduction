"""First real forward: baseline/mask/cache/depth parity and actual throughput.

Discovery only. No new-method accuracy is used to select a cache tolerance.
Cached execution is considered equivalent only at zero scalar/token error.
"""
import json
import time
from pathlib import Path
from r7_core import ROOT,read,save,sha,gated_rows


def main():
    import torch
    from runtime import load_model,encode_pair
    from ouro_cached_scoring import OuroCachedScorer
    from weight_edits import tensor_sha256
    from r7_references import arc_references,prefix_alignment
    import lm_eval
    from lm_eval.utils import load_yaml_config,apply_template
    row=gated_rows('discovery')[0];start=time.time()
    lock=read('configs/slcd_R6_FINAL_LOCK.json');assert sha('models/Ouro-1.4B/model.safetensors')==lock['model_identity']['safetensors_sha256']
    for name,value in lock['model_identity']['metadata_and_tokenizer_sha256'].items():assert sha('models/Ouro-1.4B/'+name)==value
    tok,model=load_model();scorer=OuroCachedScorer(model,5)
    head_before=tensor_sha256(model.lm_head.weight);norm_before=tensor_sha256(model.model.norm.weight)
    source_audit=read('results/slcd_SOURCE_AUDIT.json');assert head_before==source_audit['original_head_raw_tensor_sha256']
    yaml=Path(lm_eval.__file__).parent/'tasks/arc/arc_challenge.yaml';cfg=load_yaml_config(str(yaml));template=cfg['doc_to_text']
    assert isinstance(template,str)
    render=lambda doc:apply_template(template,doc)
    refs,refmeta=arc_references(row,render)
    save('configs/r7_reference_manifest.json',dict(status='frozen_before_method_feedback',references=dict(zip(['N0','N1','N2'],refs)),
        scope='ARC fixed25 protocol; placeholders replace only current question field',operation=refmeta,
        fixed_demonstrations_sha256=sha('data/slcd_fixed_demonstrations.json'),official_template=template,
        official_arc_yaml_sha256=__import__('hashlib').sha256(yaml.read_bytes()).hexdigest()))
    pairs=[encode_pair(tok,*p) for p in row['request_pairs']]
    refpairs=[[encode_pair(tok,prompt,p[1]) for p in row['request_pairs']] for prompt in refs]
    alignment=[prefix_alignment(tok,request,pair,[refpairs[j][i] for j in range(3)]) for i,(request,pair) in enumerate(zip(row['request_pairs'],pairs))]
    print('PILOT_MODEL_READY',time.time()-start,flush=True)
    torch.cuda.synchronize();begin=time.time();original=scorer.score_uncached(pairs,return_states=False);torch.cuda.synchronize();original_seconds=time.time()-begin
    old=next(x for x in __import__('r7_core').jsonlines('results/slcd_r2_discovery_development.jsonl') if x['id']==row['id'])['original_methods']
    native={f'native_k{k}':[r['sum_loglikelihood'][k] for r in original] for k in range(1,6)}
    native.update({f'loopcd_k4_{name}':[r['loopcd'][name] for r in original] for name in ['fixed','adaptive']})
    errors={key:max(abs(x-y) for x,y in zip(values,old[key]['sum_scores'])) for key,values in native.items()}
    assert max(errors.values())==0.,'Unintervened native path no longer matches original BF16 baseline'
    depth=scorer.check_depth_invariance(pairs[:1],depths=(1,4,5),atol=0.)
    print('PILOT_MAIN_PARITY',json.dumps(dict(seconds=original_seconds,max_error=max(errors.values()),depth=depth['passed'])),flush=True)
    assert depth['passed']
    # Exact original scorer check, with explicit zero tolerances.
    common=list(pairs[0][0])
    for prefix,_ in pairs[1:]:
        common=common[:next((i for i,(a,b) in enumerate(zip(common,prefix)) if a!=b),min(len(common),len(prefix)))]
    shared=scorer.build_shared_prefix(common[:-1])
    cache=scorer.compare_cached_uncached(pairs,shared=shared,char_lengths=row['choice_char_lengths'],score_atol=0.,token_atol=0.)
    print('PILOT_CACHE_PARITY',json.dumps(dict(passed=cache['passed'],max_token=max(x['max_token_abs_error'] for x in cache['detail']),max_score=max(x['score_abs_error'] for x in cache['detail']))),flush=True)
    scorer.kmax=4;scorer._set_depth(4)
    timings=[]
    for j,p in enumerate(refpairs):
        torch.cuda.synchronize();begin=time.time();scorer.score_uncached(p);torch.cuda.synchronize();timings.append(time.time()-begin)
        print('PILOT_REFERENCE_SECONDS',j,timings[-1],flush=True)
    assert tensor_sha256(model.lm_head.weight)==head_before and tensor_sha256(model.model.norm.weight)==norm_before
    output=dict(status='complete',n=1,id=row['id'],main_uncached_seconds=original_seconds,reference_uncached_seconds=timings,
                baseline_max_sum_errors=errors,independent_depth_parity=depth,cache_parity=cache,
                alignment=[dict(passed=m is not None,reason=reason,mapping=m) for m,reason in alignment],
                head_sha256=head_before,norm_sha256=norm_before,model_weights_modified=False,
                model_safetensors_sha256=lock['model_identity']['safetensors_sha256'],elapsed_seconds=time.time()-start,
                source_sha256={n:sha('scripts/'+n) for n in ['r7_pilot.py','r7_references.py','r7_core.py','ouro_cached_scoring.py','runtime.py']},
                cache_optimization_eligible=cache['passed'],cache_parity_scope='One real discovery question, all choices; further development coverage needed before deployment')
    save('results/r7_PILOT_COMPLETE.json',output);print(json.dumps({k:v for k,v in output.items() if k not in ['cache_parity','independent_depth_parity','alignment']},indent=2),flush=True)


if __name__=='__main__':main()
