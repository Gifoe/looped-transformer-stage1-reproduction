"""Actual K4 joint reference and frozen selected-only score construction."""
import numpy as np
from r10_core import joint_requests,joint_prefix
from r7_core import digest
from r11_scores import all_scores,ELIGIBLE


def infer_auxiliary(model,scorer,tok,clean,prefix,robust):
    from runtime import encode_pair
    from r7_inference import forward_states,head_logits,original_sums
    cache={};counts=[];input_counts=[];target_hash=[]
    def run(requests):
        sums={}
        for context,target in requests:
            pre,tar=encode_pair(tok,context,target);positions=tuple(range(len(pre)-1,len(pre)+len(tar)-1));key=(tuple(pre+tar[:-1]),positions)
            if key not in cache:cache[key]=head_logits(model,forward_states(scorer,(pre,tar),4))
            for name,value in original_sums(cache[key],tar).items():sums.setdefault(name,[]).append(value)
            counts.append(len(tar));input_counts.append(len(key[0]));target_hash.append(digest(tar))
        return sums
    requests=joint_requests(clean,prefix);n=len(clean['choices']);sums=run(requests)
    rotations=[dict(order=list(range(n)),sum_scores=sums,request_sha256=[digest(r) for r in requests])]
    if robust:
        for shift in range(1,n):
            order=[(j+shift)%n for j in range(n)]
            req=joint_requests(dict(question=clean['question'],choices=[clean['choices'][i] for i in order]),prefix);observed=run(req);mapped={}
            for name,vector in observed.items():
                v=np.empty(n)
                for j,i in enumerate(order):v[i]=vector[j]
                mapped[name]=v.tolist()
            rotations.append(dict(order=order,sum_scores=mapped,request_sha256=[digest(r) for r in req]))
    return dict(auxiliary_sum_scores=sums,cyclic_rotations=rotations,actual_auxiliary_depth=4,
        target_token_counts=counts,model_input_token_counts=input_counts,target_ids_sha256=target_hash,unique_auxiliary_forwards=len(cache),robust_references=robust)


def select(original,prior,aux,names):
    assert set(names)<=set(ELIGIBLE)
    values,diags=all_scores(original,prior,aux);keep=set(names)
    for n in names:keep.update([n+'_prior_norm',n+'_shuffle_update','single_native_axis' if n=='single_loop_axis' else 'robust_cyclic_native_axis'])
    return {k:values[k] for k in values if k in keep},{k:diags[k] for k in diags if k in keep}


def joint_baselines(aux):
    s=aux['auxiliary_sum_scores'];v=dict(joint1_only=s['native_k1'],joint4_only=s['native_k4'],jointFixed_only=s['loopcd_k4_fixed'])
    if aux['robust_references']:
        for depth in [1,4]:v[f'cyclic_joint{depth}_only']=np.asarray([r['sum_scores'][f'native_k{depth}'] for r in aux['cyclic_rotations']]).mean(axis=0).tolist()
    return v
