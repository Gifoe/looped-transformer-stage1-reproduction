"""Freeze source-value forecasts using three training folds and a sealed fourth fold."""

import argparse
import json
from pathlib import Path
import statistics

import torch

from collect_identification import torch_save
from task_data import hash_order
import minimal


TYPES={'arc_challenge':'science','obqa':'science','qasc':'science','scienceqa':'science',
       'csqa':'commonsense','social_iqa':'commonsense','boolq':'reading','race':'reading',
       'squad':'span','gsm8k':'math'}


def top(matrix,rank,seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return torch.svd_lowrank(matrix,q=rank,niter=7)


def axis(matrix,seed):
    u,s,w=top(matrix,12,seed)
    return (u[:,0].contiguous(),w[:,0].contiguous()),s


def projection(matrix,pair):
    u,w=pair
    return float(u@(matrix@w))


def centered_scale(records):
    values=torch.tensor([r['q_loop'] for r in records],dtype=torch.float32)
    qs=torch.quantile(values,values.new_tensor([.25,.75]))
    return max(.25,float(qs[1]-qs[0])/1.349)


def load_source(moment_root,data,source,mode,seed):
    folder=moment_root/source/mode/'k4'
    path=folder/'LOCK.json'
    spec=json.loads(path.read_text())
    input_path=data/f'{source}.jsonl'
    manifest=json.loads(input_path.with_suffix('.json').read_text())
    if spec['source_sha256']!=minimal.sha(input_path) or spec['folds']!=4 or spec['limit'] is not None:
        raise ValueError(f'Target is not an all-eligible four-fold source: {folder}')
    parts=[torch.load(folder/f'fold_{f}.pt',weights_only=True,map_location='cpu') for f in (0,1,2)]
    count=sum(x['count'] for x in parts)
    records=[r for x in parts for r in x['records']]
    if not count or not (folder/'fold_3.jsonl').exists() or manifest['retained']<=count:
        raise ValueError(f'Incomplete independent fourth fold for {folder}')
    if len(records)!=count or len({r['id'] for r in records})!=count:
        raise ValueError(f'Duplicate fitting observations for {folder}')
    sigma=centered_scale(records)
    total={name:sum((x['moments'][name] for x in parts),torch.zeros_like(parts[0]['moments'][name]))/count/sigma
           for name in parts[0]['moments']}
    total['excess']=total['loop']-total['direct']
    total['full_excess']=total['full_loop']-total['direct'] if 'full_loop' in total else total['excess']
    pair,stability={},{}
    for key in ('loop','excess'):
        pair[key],s=axis(total[key],seed)
        faxes=[]
        for fold in parts:
            m=fold['moments']['loop'] if key=='loop' else fold['moments']['loop']-fold['moments']['direct']
            definition,_=axis(m/fold['count'],seed)
            faxes.append(definition)
        product=lambda a,b:float((a[0]@b[0])*(a[1]@b[1]))
        values=[product(a,b) for i,a in enumerate(faxes) for b in faxes[:i]]
        stable=statistics.mean(values)
        energy=float(total[key].square().sum())
        stability[key]=dict(fold_pair_cosines=values,mean_cosine=stable,top_singular=s.tolist(),
                            rank1_energy=float(s[0]**2)/max(1e-12,energy),effective_stable_rank=energy/max(1e-12,float(s[0]**2)))
    difficulty=1.-statistics.mean(r['native_loop'] for r in records)
    return total,pair,stability,dict(source=source,mode=mode,n_fit=count,n_holdout=manifest['retained']-count,
                                   n_total=manifest['retained'],fit_q_robust_scale=sigma,difficulty=difficulty,
                                   fold3_access='population from input manifest; no fold-3 outcomes loaded',
                                   fit_hashes={str(f):minimal.sha(folder/f'fold_{f}.pt') for f in (0,1,2)},
                                   eligible_inputs_sha256=spec['source_sha256'])


def average(values):
    return statistics.mean(values) if values else None


def robust(values):
    return statistics.mean(values)-.5*statistics.pstdev(values) if values else None


def tangent(matrix,point):
    u,w=point
    return torch.cat((matrix@w,matrix.T@u))


def run(a):
    torch.set_num_threads(a.threads)
    domains=[tuple(tag.split(':',1)) for tag in a.sources]
    if len({name for name,_ in domains})!=len(domains) or len(domains)<3:
        raise ValueError('Use at least three distinct sources in a fixed forecast')
    matrices,pairs,diagnostics,meta={},{},{},{}
    for name,mode in domains:
        matrices[name],pairs[name],diagnostics[name],meta[name]=load_source(a.moments,a.data,name,mode,a.seed)
        print('FIT ONLY',name,meta[name]['n_fit'],meta[name]['n_holdout'],flush=True)
    t1=torch.load(a.scale_weight,weights_only=True,map_location='cpu')['state']
    amplitude=float(t1['encode.weight'].norm()*t1['out.weight'].norm())
    warm=(t1['out.weight'].flatten(),t1['encode.weight'].flatten())
    random=torch.Generator().manual_seed(a.seed)
    cold=(torch.zeros_like(warm[0]),torch.rand(warm[1].shape,generator=random).mul_(2).sub_(1)/warm[1].numel()**.5)
    tangents={name:{label:{key:tangent(values[key],point) for key in ('excess','loop')}
                    for label,point in (('warm_t1',warm),('cold',cold))} for name,values in matrices.items()}
    controls={}
    for source, definitions in pairs.items():
        for definition,pair in definitions.items():
            u,w=pair
            controls[f'{source}__{definition}']=dict(u=u.contiguous(),w=w.contiguous(),amplitude=amplitude)
    forecast=[]
    overlap={}
    for key in ('loop','excess'):
        positions=torch.stack([torch.nn.functional.normalize(matrices[s][key].flatten(),dim=0) for s in matrices])
        overlap[key]=(positions@positions.T).tolist()
    for name,source in matrices.items():
        foreign=[s for s in matrices if s!=name]
        core=sum((matrices[s]['excess']/matrices[s]['excess'].norm().clamp_min(1e-12) for s in foreign),
                 torch.zeros_like(source['excess']))/len(foreign)
        unit=core/core.norm().clamp_min(1e-12)
        shared=float((unit*source['excess']).sum())
        diagnostics[name]['shared_leave_source_out']=dict(coefficient=shared,
             relative_residual=float((source['excess']-shared*unit).norm()/source['excess'].norm().clamp_min(1e-12)))
        p_c=pairs[name]['excess']; p_m=pairs[name]['loop']
        for target in foreign:
            held=[r for r in foreign if r!=target]
            two_hold=[r for r in held if TYPES[r]!=TYPES[target]]
            for kind,refs in (('leave_source_out',held),('leave_task_type_out',two_hold)):
                if not refs:
                    continue
                ordinary=[projection(matrices[r]['loop'],p_c) for r in refs]
                ordinary_policy=[projection(matrices[r]['loop'],p_m) for r in refs]
                gradient=[float(torch.nn.functional.cosine_similarity(source['loop'].flatten(),matrices[r]['loop'].flatten(),dim=0)) for r in refs]
                excess=[projection(matrices[r]['excess'],p_c) for r in refs]
                absolute=[projection(matrices[r]['loop'],p_c) for r in refs]
                # The minimum gate prevents credit when the direct term alone deteriorates.
                coupled=[min(a,b) for a,b in zip(excess,absolute,strict=True)]
                first_step={}
                for init in ('warm_t1','cold'):
                    g=tangents[name][init]['loop']
                    transfer=[min(float(g@tangents[r][init]['excess']),float(g@tangents[r][init]['loop'])) for r in refs]
                    first_step[f'loop_factor_influence_{init}']=robust(transfer)
                forecast.append(dict(source=name,target=target,scope=kind,reference_sources=refs,
                             predicted=dict(random=int(hash_order(f'{name}/{target}/{kind}','loop-identification/random'),16)/(2**256),
                               difficulty=meta[name]['difficulty'],gradient_alignment=average(gradient),
                               source_only=diagnostics[name]['excess']['mean_cosine']*diagnostics[name]['excess']['top_singular'][0],
                               ordinary_functional=robust(ordinary),ordinary_policy=robust(ordinary_policy),loop_contrast=robust(excess),
                               loop_and_absolute=robust(coupled)*max(0.,diagnostics[name]['excess']['mean_cosine']),**first_step)))
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'FORECAST.json').exists():
        raise ValueError('A forecast has already been sealed at this path')
    torch_save(a.out/'DIRECTIONS.pt',controls)
    plan=dict(records=forecast,sources=meta,source_order=list(matrices),alignment=overlap,spectra=diagnostics,seed=a.seed,
              temperature='16 log score per character or per-question average teacher-forced token margin',
              forecast_target='mean finite positive absolute loop gain with separately reported loop interaction and accuracy',
              scale_weight_sha256=minimal.sha(a.scale_weight),directions_sha256=minimal.sha(a.out/'DIRECTIONS.pt'),
              code_sha256=minimal.sha(__file__))
    minimal.dump(a.out/'FORECAST.json',plan)
    print('SEALED',len(forecast),minimal.sha(a.out/'FORECAST.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('moments','data','out','scale-weight'):
        p.add_argument(f'--{field}',type=Path,required=True)
    p.add_argument('--sources',nargs='+',required=True)
    p.add_argument('--seed',type=int,default=261125137)
    p.add_argument('--threads',type=int,default=4)
    run(p.parse_args())
