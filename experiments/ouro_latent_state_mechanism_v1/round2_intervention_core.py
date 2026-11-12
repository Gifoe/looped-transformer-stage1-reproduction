"""Full-sequence ROUND 2 interventions. Requires externally validated run gates."""
import time,torch
from round2_full_sequence import full_boundary,rescale_direction,compose_state

CORE=('K2C_K2R','K1C_K2R','K2C_K1R','K1C_K1R')
NORM=('C_K1DIR_K2NORM','R_K1DIR_K2NORM','C_K2DIR_K1NORM','R_K2DIR_K1NORM')

@torch.inference_mode()
def run_question(model,records,gold):
    """Exact same batch-one production scorer for each condition, no labels into model."""
    started=time.perf_counter();records=sorted(records,key=lambda r:r['candidate'])
    if len(records)<2 or [r['candidate'] for r in records]!=list(range(len(records))):raise ValueError('candidate completeness')
    ids=records[0]['encoding']['input_ids'];pos=records[0]['encoding']['predecessor_positions']
    if not pos or any(r['encoding']['input_ids']!=ids or r['encoding']['predecessor_positions']!=pos or len(r['encoding']['target_ids'])!=1 for r in records):raise ValueError('nonshared input or multi-token continuation unsupported')
    if len(pos)!=1 or not 0<=gold<len(records):raise ValueError('invalid readout or gold')
    targets=[r['encoding']['target_ids'][0] for r in records];x=torch.tensor([ids],dtype=torch.long,device=model.lm_head.weight.device)
    def forward(depth,**opts):
        norm=[]
        def hook(module,args,out):
            if not torch.is_tensor(out) or out.ndim!=3:raise RuntimeError('invalid norm output')
            norm.append(out[0,pos,:].detach().clone())
        h=model.model.norm.register_forward_hook(hook)
        try:
            with full_boundary(model,depth,**opts) as states:y=model(x,labels=None,use_cache=False,exit_at_step=depth-1)
        finally:h.remove()
        if len(norm)!=depth:raise RuntimeError('norm count')
        z=model.lm_head(norm[-1].unsqueeze(0))[0];native=y.logits[0,pos,:].detach().clone();del y
        if z.dtype!=torch.bfloat16 or native.dtype!=torch.bfloat16 or not torch.isfinite(z).all():raise RuntimeError('bad logits')
        if not torch.equal(z,native):raise RuntimeError('production/full-head zero-tolerance bridge failed')
        scores=[]
        for target in targets:
            t=torch.tensor([target],device=z.device,dtype=torch.long);scores.append(float(z.float().log_softmax(-1).gather(-1,t[:,None]).sum()))
        return states,scores,z.detach().clone()
    s1,base1,z1=forward(1);k1=s1[0];s2,base2,z2=forward(2);k2=s2[1]
    if base1!=[r['scores'][0] for r in records] or base2!=[r['scores'][1] for r in records]:raise RuntimeError('baseline mismatch against accepted cohort')
    # First-index tie breaking, excluding gold; fixed from native K1 before interventions.
    wrong=max((i for i in range(len(targets)) if i!=gold),key=lambda i:(base1[i],-i))
    def pack(scores):
        prediction=max(range(len(scores)),key=lambda i:(scores[i],-i));return {'scores':scores,'prediction':prediction,'top_tie':sum(v==scores[prediction] for v in scores)>1,'gold_vs_fixed_wrong_gap':scores[gold]-scores[wrong]}
    conditions={};logits={'baseline_K1':z1.detach().cpu().contiguous(),'baseline_K2':z2.detach().cpu().contiguous()}
    for name in CORE:
        _,scores,z=forward(2,condition=name,k1=k1,k2_reference=k2,readout=pos)
        if name=='K2C_K2R' and (scores!=base2 or not torch.equal(z,z2)):raise RuntimeError('K2 identity anchor failure')
        if name=='K1C_K1R' and (scores!=base1 or not torch.equal(z,z1)):raise RuntimeError('K1 full replacement anchor failure')
        conditions[name]=pack(scores);logits['core_'+name]=z.detach().cpu().contiguous()
    # Pre-registered controls cover both regions, all questions. Preserve actual BF16 norms.
    same,_=rescale_direction(k2,k2)
    if not torch.equal(same,k2):raise RuntimeError('rescale(K2,K2) is not bitwise identity')
    del same
    n12,stats12=rescale_direction(k1,k2);n21,stats21=rescale_direction(k2,k1)
    norm_conditions={};mask=torch.ones(k2.shape[1],dtype=torch.bool,device=k2.device);mask[pos]=False
    for name,donor,mode in [('C_K1DIR_K2NORM',n12,'K1C_K2R'),('R_K1DIR_K2NORM',n12,'K2C_K1R'),('C_K2DIR_K1NORM',n21,'K1C_K2R'),('R_K2DIR_K1NORM',n21,'K2C_K1R')]:
        mixed=compose_state(donor,k2,pos,mode)
        if mode=='K1C_K2R':outside_ok=torch.equal(mixed[:,pos,:],k2[:,pos,:]);inside_ok=torch.equal(mixed[:,mask,:],donor[:,mask,:])
        else:outside_ok=torch.equal(mixed[:,mask,:],k2[:,mask,:]);inside_ok=torch.equal(mixed[:,pos,:],donor[:,pos,:])
        if not outside_ok or not inside_ok or not torch.isfinite(mixed).all():raise RuntimeError('norm control region/finite mismatch')
        del mixed
        _,scores,z=forward(2,condition=mode,k1=donor,k2_reference=k2,readout=pos);norm_conditions[name]=pack(scores);logits['norm_'+name]=z.detach().cpu().contiguous()
    # token-local norm/head: final-layer context-only replacement must not affect readout.
    l1,_,zz1=forward(1,layer_index=23);l2,_,zz2=forward(2,layer_index=23)
    if not torch.equal(zz1,z1) or not torch.equal(zz2,z2):raise RuntimeError('last-layer capture changed baseline')
    _,neg,zn=forward(2,condition='K1C_K2R',k1=l1[0],k2_reference=l2[1],readout=pos,layer_index=23)
    if neg!=base2 or not torch.equal(zn,z2):raise RuntimeError('last-layer C-only negative control failed')
    logits['negative_last_C']=zn.detach().cpu().contiguous()
    def normdata(d):
        if any(not torch.isfinite(v).all() for v in d.values()):raise RuntimeError('nonfinite measured norm')
        return {**{k:v.tolist() for k,v in d.items()},'max_abs_actual_minus_reference':float((d['actual_bf16_norm']-d['reference_norm']).abs().max())}
    result={'gold':gold,'k1_highest_wrong':wrong,'readout_positions':pos,'context_positions':[i for i in range(len(ids)) if i not in set(pos)],'context_positions_count':len(ids)-len(set(pos)),'input_length':len(ids),'target_ids':targets,'baseline_K1':pack(base1),'baseline_K2':pack(base2),'conditions':conditions,'norm_conditions':norm_conditions,'norms':{'K1DIR_K2NORM':normdata(stats12),'K2DIR_K1NORM':normdata(stats21)},'controls':{'K2_identity':'exact','K1_full_anchor':'exact','last_layer_C_only':'exact','rescale_K2_to_K2':'exact'},'elapsed_seconds':time.perf_counter()-started}
    return result,logits
