"""CPU-only deterministic 14-candidate construction from COMPLETE training stats.

No selection/validation/test scores are opened. Candidate direction construction
uses gold-margin-conditioned forward means, not label-blind PCA or optimization.
Frozen factors are authoritative; SVD recomputation across libraries is not a
bit-exact replay mechanism. Rank ties use physical layer / input column indices.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import torch
from safetensors import safe_open
from weight_edits import ROOT, REVISION, file_sha256, tensor_sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    tmp.replace(path)


def write_csv(path, rows):
    with path.open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def sign_fix(U):
    U = U.clone()
    for j in range(U.shape[1]):
        if U[torch.argmax(U[:, j].abs()), j] < 0:
            U[:, j].neg_()
    return U


def make_groups(ids, scores, transitions, seed):
    """Fixed quartiles across all 384 questions, including all correctness types."""
    result = []
    for a, b in transitions:
        delta = []
        counts = {'C_to_W': 0, 'W_to_C': 0, 'C_to_C': 0, 'W_to_W': 0}
        categories = []
        for qid in ids:
            before = scores[qid]['methods'][f'native_k{a}']
            after = scores[qid]['methods'][f'native_k{b}']
            change = float(after['margin'])-float(before['margin'])
            require(math.isfinite(change), 'Nonfinite margin')
            delta.append(change)
            category = ('C' if before['correct'] else 'W')+'_to_'+('C' if after['correct'] else 'W')
            categories.append(category); counts[category] += 1
        order = sorted(range(len(ids)), key=lambda i: (delta[i], hashlib.sha256(
            f'V2-trend|{seed}|{a}to{b}|{ids[i]}'.encode()).hexdigest()))
        lo, hi = order[:96], order[-96:]
        result.append(dict(transition=[a,b], bottom_indices=lo, top_indices=hi,
                           bottom_ids=[ids[i] for i in lo], top_ids=[ids[i] for i in hi],
                           transition_category_counts=counts,
                           bottom_category_counts={c:sum(categories[i]==c for i in lo) for c in counts},
                           top_category_counts={c:sum(categories[i]==c for i in hi) for c in counts},
                           margin_changes=[dict(id=qid, delta=delta[i], category=categories[i]) for i,qid in enumerate(ids)]))
    return result


def site_direction(values, groups, weights):
    """values [question,loop,output]; FP64 reductions over stored FP32 means."""
    dy = values.double().diff(dim=1)
    D = torch.zeros(dy.shape[-1], dtype=torch.float64)
    energy = 0.
    for t, (g,w) in enumerate(zip(groups,weights)):
        D += w*(dy[g['bottom_indices'],t].mean(0)-dy[g['top_indices'],t].mean(0))
        energy += w*float(dy[:,t].square().sum(-1).mean())
    norm = float(torch.linalg.vector_norm(D))
    return D, dict(direction_norm=norm, transition_rms=math.sqrt(energy),
                   score=norm/(math.sqrt(energy)+1e-12), nondegenerate=norm>1e-12 and energy>0)


def construct_bases(values, D, groups, weights):
    u = sign_fix((D/torch.linalg.vector_norm(D))[:,None])
    dy = values.double().diff(dim=1)
    blocks = []
    for t,(g,w) in enumerate(zip(groups,weights)):
        x = dy[g['bottom_indices'],t]
        x = x-x.mean(0)
        x = x-(x@u)@u.T
        blocks.append(x*math.sqrt(w/96))
    x = torch.cat(blocks,0)
    _, singular, vh = torch.linalg.svd(x, full_matrices=False)
    tolerance = float(singular[0])*max(x.shape)*torch.finfo(torch.float64).eps
    require(len(singular)>=3 and float(singular[2])>tolerance,
            'Rank4 direction has fewer than three nonzero residual PCs; no random filler allowed')
    q,_ = torch.linalg.qr(torch.cat([u,vh[:3].T],1), mode='reduced')
    U1, U4 = u.float().contiguous(), sign_fix(q).float().contiguous()
    for U in (U1,U4):
        require(torch.allclose(U.T@U,torch.eye(U.shape[1]),atol=1e-5,rtol=1e-5), 'Invalid basis Gram matrix')
    require(float((U4[:,0]-U1[:,0]).abs().max())<1e-6, 'Rank4 lost rank1 direction')
    return U1,U4,dict(singular_values=singular.tolist(), numerical_rank_tolerance=tolerance,
                       numerical_rank=int((singular>tolerance).sum()),
                       rank4_gram_max_error=float((U4.T@U4-torch.eye(4)).abs().max()))


def rank_columns(a2, weight, u, groups, weights):
    dy = a2.double().diff(dim=1)
    contrast = torch.zeros(a2.shape[-1],dtype=torch.float64)
    for t,(g,w) in enumerate(zip(groups,weights)):
        contrast += w*(dy[g['bottom_indices'],t].mean(0)-dy[g['top_indices'],t].mean(0))
    W = weight.double(); alignment=(u.double().T@W).squeeze(0).square()
    scores=contrast.clamp_min(0)*alignment
    fallback=bool(torch.count_nonzero(scores)==0)
    energy=a2.double().mean((0,1))*W.square().sum(0)
    if fallback: scores=energy
    require(torch.isfinite(scores).all() and torch.count_nonzero(scores)>0, 'No nonzero column ranking')
    order=sorted(range(len(scores)),key=lambda j:(-float(scores[j]),j))
    rows=[dict(rank=r+1,column=j,score=float(scores[j]),contrast=float(contrast[j]),
               writer_alignment_squared=float(alignment[j]),activation_weight_energy=float(energy[j]),
               all_zero_fallback=fallback) for r,j in enumerate(order)]
    return order,rows,fallback


@torch.no_grad()
def main():
    torch.set_num_threads(4); torch.set_grad_enabled(False)
    design_path=ROOT/'configs/CANDIDATE_DESIGN.json'; design=read_json(design_path)
    protocol=read_json(ROOT/'configs/protocol.json')
    require(design['candidate_count']==14 and design['discovery_n']==384, 'Unexpected design')
    require(protocol['candidate_design_sha256']==file_sha256(design_path), 'Protocol/design hash mismatch')
    require(design['model_revision']==REVISION, 'Wrong checkpoint revision')
    out=ROOT/'edited_weights/statistics'; frozen_path=ROOT/'configs/CANDIDATES_FROZEN.json'
    stats=ROOT/'results/training_statistics'
    complete=read_json(stats/'COMPLETE_384.json')
    require(complete['status']=='complete' and complete['n']==384,'Require completed 384-question collection')
    lock=read_json(ROOT/'configs/DATA_LOCK.json')
    require(lock['discovery_selection_demos_disjoint'] and not lock['validation_or_test_opened'], 'Invalid data lock')
    parity=read_json(stats/'PARITY.json')
    require(parity['status']=='passed' and parity['exact_recording_and_v1_original'] and len(parity['cases'])>=2,
            'Require two exact native recording parity cases')
    require(all(v==0 for c in parity['cases'] for v in c['errors'].values()),'Nonzero recording parity error')
    manifest_path=ROOT/'data/discovery_expanded.jsonl'
    require(lock['cohorts']['discovery_expanded']['n']==384 and
            lock['cohorts']['discovery_expanded']['sha256']==file_sha256(manifest_path),
            'Discovery manifest differs from DATA_LOCK')
    rows=[json.loads(line) for line in manifest_path.read_text(encoding='utf-8-sig').splitlines() if line]
    ids=[row['id'] for row in rows]
    require(len(ids)==len(set(ids))==384,'Training manifest must have 384 unique IDs')
    require(all(qid.startswith('arc_challenge/train/') for qid in ids),'Non-training discovery ID')
    score_rows=[json.loads(line) for line in (stats/'scores.jsonl').read_text().splitlines() if line]
    scores={row['id']:row for row in score_rows}
    require(len(score_rows)==len(scores)==384 and set(scores)==set(ids),'Incomplete or extra training scores')
    for row in rows:
        score=scores[row['id']]
        require(score['gold']==row['gold'] and score['choice_char_lengths']==row['choice_char_lengths'],
                'Training candidate labels or normalization lengths changed')
        for k in range(1,7):
            method=score['methods'][f'native_k{k}']
            normalized=method['normalized_scores']; gold=row['gold']
            require(len(normalized)==len(method['sum_scores'])==len(row['choice_char_lengths']) and len(normalized)>1,
                    'Invalid candidate scores')
            require(all(math.isfinite(x) for x in normalized),'Nonfinite normalized score')
            require(all(x==s/max(n,1) for x,s,n in zip(normalized,method['sum_scores'],row['choice_char_lengths'])),
                    'Native character-normalization mismatch')
            pred=max(range(len(normalized)),key=lambda j:normalized[j])
            margin=normalized[gold]-max(x for j,x in enumerate(normalized) if j!=gold)
            require(method['prediction']==pred and method['correct']==(pred==gold) and method['margin']==margin,
                    'Native margin/prediction receipt mismatch')
    provenance=dict(design_sha256=file_sha256(design_path), manifest_sha256=file_sha256(manifest_path),
                    scores_sha256=file_sha256(stats/'scores.jsonl'), complete_sha256=file_sha256(stats/'COMPLETE_384.json'),
                    data_lock_sha256=file_sha256(ROOT/'configs/DATA_LOCK.json'), parity_sha256=file_sha256(stats/'PARITY.json'),
                    discovery_ids_sha256=digest(ids), implementation_sha256=complete['implementation_sha256'],
                    builder_sha256=file_sha256(Path(__file__)), model_revision=REVISION,
                    candidate_io_sha256=file_sha256(ROOT/'scripts/candidate_io.py'),
                    weight_edits_sha256=file_sha256(ROOT/'scripts/weight_edits.py'),
                    labels_used='training gold margins only; not label-blind', statistics_files=[])
    # Existing frozen artifacts are immutable. Repeated invocation verifies receipts.
    if frozen_path.exists():
        old=read_json(frozen_path)
        for key,value in provenance.items():
            if key!='statistics_files': require(old['provenance'][key]==value,'Frozen provenance changed: '+key)
        for item in old['provenance']['statistics_files']:
            require(file_sha256(stats/item['file'])==item['sha256'],'Frozen statistics changed')
        for rel,sha in old['artifact_sha256'].items(): require(file_sha256(ROOT/rel)==sha,'Frozen artifact changed: '+rel)
        print('CANDIDATES_ALREADY_FROZEN_VERIFIED',flush=True); return
    require(not out.exists() or not any(out.iterdir()),'Partial statistics outputs exist; inspect before rebuilding')
    # These are metadata-only existence checks, never held-out content reads.
    for pattern in ('*selection*/scores.jsonl','*validation*/scores.jsonl','*test*/scores.jsonl'):
        require(not list((ROOT/'results').glob(pattern)),'Candidate freeze must precede held-out outcome files')
    groups=make_groups(ids,scores,design['trends']['transitions'],design['seed'])
    weights=design['trends']['weights']
    outputs={b:torch.empty((384,24,4,2048),dtype=torch.float32) for b in ('mlp','attention')}
    paths=[]
    for i,row in enumerate(rows):
        score=scores[row['id']]
        require(score['scope']=='train_discovery_only' and score['gold']==row['gold'] and
                score['prompt_sha256']==row['prompt_sha256'] and score['implementation_sha256']==complete['implementation_sha256'], 'Score provenance mismatch')
        path=(stats/score['statistics_file']).resolve()
        require(path.parent==stats.resolve(),'Unexpected statistics path')
        require(path not in paths,'Duplicate statistics filename'); paths.append(path)
        payload=torch.load(path,map_location='cpu',weights_only=True)
        require(payload['id']==row['id'] and payload['loops']==[3,4,5,6] and payload['implementation_sha256']==complete['implementation_sha256'],'Payload provenance mismatch')
        require(set(payload['statistics'])=={b+s for b in outputs for s in ('_input_second_moment','_raw_output_mean')},'Unexpected statistics keys')
        for b in outputs:
            for suffix,width in (('_input_second_moment',5632 if b=='mlp' else 2048),('_raw_output_mean',2048)):
                x=payload['statistics'][b+suffix]
                require(x.dtype==torch.float32 and list(x.shape)==[24,4,width] and torch.isfinite(x).all(),'Invalid statistics shape/dtype/values')
                if suffix=='_input_second_moment': require((x>=0).all(),'Negative second moment')
            outputs[b][i].copy_(payload['statistics'][b+'_raw_output_mean'])
        provenance['statistics_files'].append(dict(id=row['id'],file=path.name,sha256=file_sha256(path)))
        del payload
    ranked={}; sites=[]; directions={}
    for b,values in outputs.items():
        branch=[]
        for layer in range(24):
            D,summary=site_direction(values[:,layer],groups,weights)
            directions[b,layer]=D
            branch.append(dict(branch=b,physical_layer=layer,**summary))
        ranked[b]=sorted([s for s in branch if s['nondegenerate']],key=lambda s:(-s['score'],s['physical_layer']))
        sites.extend(branch)
    require(len(ranked['mlp'])>=2 and len(ranked['attention'])>=1,'Too few nondegenerate sites')
    selected={'mlp_first':ranked['mlp'][0], 'mlp_second':ranked['mlp'][1], 'attention_first':ranked['attention'][0]}
    checkpoint=ROOT/'models/Ouro-1.4B/model.safetensors'
    require(read_json(ROOT/'models/Ouro-1.4B/snapshot_metadata.json')['sha']==REVISION,'Snapshot revision mismatch')
    verified=read_json(ROOT/'logs/MODEL_SERVER_VERIFIED.json')
    require(verified['revision']==REVISION and verified['status']=='passed','Missing verified model receipt')
    provenance['checkpoint_file_sha256']=file_sha256(checkpoint)
    provenance['model_verified_receipt_sha256']=file_sha256(ROOT/'logs/MODEL_SERVER_VERIFIED.json')
    bases={}; details={}; columns={}; csvs={}
    with safe_open(checkpoint,framework='pt',device='cpu') as tensors:
        for label,site in selected.items():
            b,l=site['branch'],site['physical_layer']
            name=f'model.layers.{l}.'+('mlp.down_proj' if b=='mlp' else 'self_attn.o_proj')+'.weight'
            W=tensors.get_tensor(name)
            require(W.dtype==torch.bfloat16 and list(W.shape)==[2048,5632 if b=='mlp' else 2048],'Checkpoint projection dimensions/dtype mismatch')
            U1,U4,diagnostics=construct_bases(outputs[b][:,l],directions[b,l],groups,weights)
            bases[label+'_rank1']=U1; bases[label+'_rank4']=U4
            details[label]=dict(**site,name=name,base_tensor_sha256=tensor_sha256(W), basis_diagnostics=diagnostics)
            if b=='mlp':
                a2=torch.stack([torch.load(p,map_location='cpu',weights_only=True)['statistics']['mlp_input_second_moment'][l] for p in paths])
                order,table,fallback=rank_columns(a2,W,U1,groups,weights)
                columns[label]=order; csvs[label+'_column_scores.csv']=table
                details[label]['column_score_all_zero_fallback']=fallback
    out.mkdir(parents=True,exist_ok=True)
    directions_rel='edited_weights/statistics/bases.pt'
    torch.save(bases,ROOT/directions_rel)
    direction_sha=file_sha256(ROOT/directions_rel); provenance_sha=digest(provenance)
    concrete=[]
    for recipe in design['candidates']:
        edit_list=[]; multi='sites' in recipe
        for label in (recipe['sites'] if multi else [recipe['site']]):
            detail=details[label]; attenuation=recipe['mode']=='sparse_column_attenuation'
            rank=recipe.get('rank_each' if multi else 'rank')
            count=recipe.get('columns_each' if multi else 'columns')
            key=None if attenuation else label+f'_rank{rank}'
            edit_list.append(dict(name=detail['name'],mode='attenuation' if attenuation else 'projection',
                alpha=recipe['alpha_each' if multi else 'alpha'], columns=None if count is None else columns[label][:count],
                basis_key=key, basis_sha256=None if key is None else tensor_sha256(bases[key]),
                U_file=None if key is None else directions_rel,U_key=key,
                base_tensor_sha256=detail['base_tensor_sha256'], physical_layer=detail['physical_layer']))
        concrete.append(dict(id=recipe['id'],family=recipe['family'],description=recipe['mode'],edits=edit_list,
                             directions_file=directions_rel,directions_sha256=direction_sha,
                             discovery_provenance_sha256=provenance_sha))
    require(len(concrete)==14 and len({c['id'] for c in concrete})==14,'Candidate count mismatch')
    write_csv(out/'site_scores.csv',sorted(sites,key=lambda s:(s['branch'],-s['score'],s['physical_layer'])))
    for name,table in csvs.items(): write_csv(out/name,table)
    write_json(out/'groups.json',dict(tie_break_serialization='V2-trend|2021|{a}to{b}|{question_id}',groups=groups))
    write_json(out/'selected_sites.json',details)
    write_json(out/'provenance.json',provenance)
    artifacts={str(p.relative_to(ROOT)).replace('\\','/'):file_sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(frozen_path,dict(status='frozen_before_candidate_evaluation',candidate_count=14,
        model_revision=REVISION, candidates=concrete, directions_file=directions_rel,
        directions_sha256=direction_sha,provenance=provenance,discovery_provenance_sha256=provenance_sha,
        artifact_sha256=artifacts,selection_order=design['selection_order'],
        no_validation_or_test_direction_updates=True,baseline_eligible_for_final_selection=False,
        selected_sites={k:v['physical_layer'] for k,v in selected.items()},
        interpretation='Exploratory label-conditioned training directions; no causal-direction or held-out benefit claim.'))
    print(json.dumps(dict(status='frozen',candidates=14,selected_sites={k:v['physical_layer'] for k,v in selected.items()},
                          frozen_sha256=file_sha256(frozen_path))),flush=True)


if __name__=='__main__': main()
