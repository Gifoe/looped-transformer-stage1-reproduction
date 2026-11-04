"""Shared joint inference for the selected-only held-out execution path.

No gold accepted. Complete letter targets and exact duplicate inputs only.
"""
import numpy as np
from r10_core import digest,joint_requests


def infer(model,scorer,tok,clean,prefix,balanced):
    from runtime import encode_pair
    from r7_inference import forward_states,head_logits,original_sums
    cache={};counts=[];targets=[];inputs=[]
    def run(requests):
        result={}
        for request in requests:
            pre,tar=encode_pair(tok,*request);positions=tuple(range(len(pre)-1,len(pre)+len(tar)-1))
            key=(tuple(pre+tar[:-1]),positions)
            if key not in cache:cache[key]=head_logits(model,forward_states(scorer,(pre,tar),5))
            for name,value in original_sums(cache[key],tar).items():result.setdefault(name,[]).append(value)
            counts.append(len(tar));targets.append(digest(tar));inputs.append(len(key[0]))
        return result
    requests=joint_requests(clean,prefix);sums=run(requests);n=len(clean['choices'])
    rotations=[dict(order=list(range(n)),sum_scores=sums,request_sha256=[digest(p) for p in requests])]
    if balanced:
        for shift in range(1,n):
            order=[(i+shift)%n for i in range(n)];rotated=dict(question=clean['question'],choices=[clean['choices'][i] for i in order])
            rq=joint_requests(rotated,prefix);observed=run(rq);mapped={}
            for name,vector in observed.items():
                scores=np.empty(n)
                for j,i in enumerate(order):scores[i]=vector[j]
                mapped[name]=scores.tolist()
            rotations.append(dict(order=order,sum_scores=mapped,request_sha256=[digest(p) for p in rq]))
    b4=np.mean([r['sum_scores']['native_k4'] for r in rotations],axis=0) if balanced else None
    return dict(auxiliary_sum_scores=sums,cyclic_rotations=rotations,balanced_native4_scores=None if b4 is None else b4.tolist(),
        target_token_counts=counts,target_ids_sha256=targets,model_input_token_counts=inputs,unique_auxiliary_forwards=len(cache))


def selected_scores(base,aux,names):
    from r10_scores import score,ELIGIBLE
    names=list(names);assert set(names)<=set(ELIGIBLE)
    s=aux['auxiliary_sum_scores'];b=aux['balanced_native4_scores']
    assert 'balanced4_protected' not in names or b is not None
    # Unselected ranking vectors remain local and have no evaluated outcomes,
    # saved predictions or correctness. The immutable math is shared exactly.
    allscores,diags=score(base['A1'],base['M0_R6'],base['fixed'],s['native_k4'],s['loopcd_k4_fixed'],s['native_k4'] if b is None else b)
    controls={('balanced4' if n=='balanced4_protected' else 'jointFixed' if n=='jointFixed_protected' else 'joint4')+'_shuffled_control' for n in names if n!='joint4_fusion'}
    values={n:allscores[n] for n in names+sorted(controls)}
    if 'joint4_fusion' in names:
        from r9_scores import evidence
        r=np.asarray(base['M0_R6']);u,_=evidence(r,s['native_k4']);p=np.random.default_rng(2021).permutation(len(r))
        values['joint4_raw_shuffled_control']=r+u[p]
    diagkeys={('balanced4' if n=='balanced4_protected' else 'jointFixed' if n=='jointFixed_protected' else 'joint4') for n in names}
    return values,{k:diags[k] for k in diagkeys}


def baselines(aux):
    s=aux['auxiliary_sum_scores'];q4=np.asarray(s['native_k4']);q5=np.asarray(s['native_k5']);f=np.asarray(s['loopcd_k4_fixed'])
    result=dict(joint4_only=q4,jointFixed_only=f,jointA1_only=f+q4-q5)
    if aux['balanced_native4_scores'] is not None:result['balanced4_only']=np.asarray(aux['balanced_native4_scores'])
    return result
