"""Small nl33w ARC causal audit. Case selection and controls precede interventions.

Sixteen unique questions: two per receiver for harm/rescue crossed with whether
the CUT partner was correct. New microbatch ON/CUT replay is recorded explicitly.
No nl23 question partition is reused; no new model or large training is involved.
"""
import argparse
import hashlib
import json
import time
from linux_runtime import ROOT,load_frozen,save


def select_cases():
    src=ROOT/'fresh/arc_first'
    d=json.loads((src/'items.json').read_text());items=d['items'];n=len(items)
    scores={c:{x:[None]*n for x in ['quill','ember']} for c in ['link','cut']}
    for c in scores:
        for p in src.glob('batch_*_'+c+'.json'):
            r=json.loads(p.read_text())
            for x in scores[c]:
                for i,row in zip(r['question_indices'],r['per_receiver'][x]):scores[c][x][i]=row
    used=set();cases=[];candidates={}
    for x in ['quill','ember']:
        y='ember' if x=='quill' else 'quill'
        groups={k:[] for k in ['harm_both_cut_correct','harm_partner_cut_wrong','rescue_partner_cut_correct','rescue_both_cut_wrong']}
        for i in range(n):
            before=scores['cut'][x][i]['correct'];after=scores['link'][x][i]['correct'];peer=scores['cut'][y][i]['correct']
            if before and not after:groups['harm_both_cut_correct' if peer else 'harm_partner_cut_wrong'].append(i)
            if not before and after:groups['rescue_partner_cut_correct' if peer else 'rescue_both_cut_wrong'].append(i)
        for category,ids in groups.items():
            rank=sorted(ids,key=lambda i:hashlib.sha256(('nl33w-causal-pilot-20261030|'+x+'|'+category+'|'+d['original_ids'][i]).encode()).hexdigest())
            candidates[x+'/'+category]=rank
            chosen=[i for i in rank if i not in used][:2]
            assert len(chosen)==2,(x,category,'insufficient unique cases')
            for i in chosen:
                cases.append(dict(question_index=i,question_id=d['original_ids'][i],target_receiver=x,category=category,
                    item=items[i],saved_baseline={c:{z:scores[c][z][i] for z in scores[c]} for c in scores}))
                used.add(i)
    return cases,candidates


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bs',type=int,default=2)
    ap.add_argument('--selection-file');ap.add_argument('--study',default='causal_pilot_16')
    ap.add_argument('--dot64',action='store_true');ap.add_argument('--dissection',action='store_true');args=ap.parse_args()
    assert args.bs>=2
    assert args.study.replace('_','').isalnum()
    out=ROOT/'fresh'/args.study;out.mkdir(exist_ok=True)
    prescribed=json.loads((ROOT/args.selection_file).read_text()) if args.selection_file else None
    cases,candidates=(prescribed['cases'],prescribed.get('candidate_order',{})) if prescribed else select_cases()
    selection=dict(checkpoint='released_final_nl33w',source='Complete fresh1165 ARC ON/CUT',
        rule=prescribed['rule'] if prescribed else 'Two unique cases per receiver/category, SHA-256 order independent of logit margin; no replacements after intervention outcomes',
        cases=cases,candidate_order=candidates,bs=args.bs,prescribed_plan=prescribed,
        caveat='Selected transitions are a diagnostic case series, not a population accuracy estimate. New batch-shape replay instability is reported.')
    if (out/'SELECTION.json').exists():assert json.loads((out/'SELECTION.json').read_text())==selection
    else:save(out/'SELECTION.json',selection)
    start=time.time();save(out/'WORKER.json',dict(status='loading',started_unix=start))
    pair,toks=load_frozen()
    import torch
    import session as S
    import link
    from causal_trace import CausalTrace
    from mcq_tools import prepare,letter_logits,outcomes,frozen_readout_rows,add_dot64_readout
    readouts=frozen_readout_rows(pair) if args.dot64 else None
    # Only length sorting; question selection has already been frozen to disk.
    if prescribed:
        bounds=[];p=0
        for length in prescribed['batch_lengths']:
            bounds.append((p,p+length));p+=length
        assert p==len(cases)
    else:
        cases=sorted(cases,key=lambda r:sum(len(str(r['item'][k])) for k in ['q','choices']))
        bounds=list(S.batches(len(cases),args.bs))
    with torch.inference_mode():
        for bid,(a,z) in enumerate(bounds):
            path=out/f'batch_{bid:03d}.json'
            if path.exists():continue
            chosen=cases[a:z];items=[r['item'] for r in chosen]
            b,own,built=prepare(pair,items)
            positions={}
            for x in pair.models:
                y=link.partner(x);rows=[]
                for i in range(len(items)):
                    tick=b[y]['tick'][i,int(own[y][i,0])]
                    visible=(b[x]['tick'][i]<=tick)&b[x]['att'][i].bool()
                    last=int(visible.nonzero()[-1]);assert last<=int(own[x][i,0]),'sender gold letter would enter probe'
                    rows.append([int(own[x][i,0]),last])
                positions[x]=torch.tensor(rows,device=b[x]['ids'].device)
            records={};traces={};timing={}
            for name,kw in [('cut',dict(keep_receivers=[])),('link',{})]:
                before=time.time()
                with CausalTrace(pair,positions,capture_full=True,**kw) as tr:
                    last,_=pair.joint(b)
                lg=letter_logits(pair,last,own);records[name]=outcomes(lg,items);traces[name]=tr
                if readouts:add_dot64_readout(records[name],last,own,readouts)
                for x in pair.models:
                    tr.states[(x,35,'final_normalized')]=tr.select(x,last[x]).detach().float().cpu()
                torch.save(tr.states,out/f'batch_{bid:03d}_{name}_states.pt')
                timing[name]=time.time()-before
            if prescribed:
                for i,case in enumerate(chosen):
                    for cond in ['link','cut']:
                        for x in pair.models:
                            assert records[cond][x][i]['letter_logits']==case['saved_baseline'][cond][x]['letter_logits'],('Original-batch replay mismatch',case['question_id'],cond,x)
            intervention_specs=[]
            intervention_specs += [('ember_to_quill_only',dict(keep_receivers=['quill'])),('quill_to_ember_only',dict(keep_receivers=['ember']))]
            intervention_specs += [('drop_layer_'+str(l),dict(keep_blocks=[k for k in pair.xblocks if k!=l])) for l in pair.xblocks]
            intervention_specs += [('write_scale_0.5',dict(scale=.5)),('write_scale_1.5',dict(scale=1.5))]
            for x in pair.models:
                intervention_specs.append(('clean_sender_'+x,dict(sender_donors={(x,l):traces['cut'].full_pre[(x,l)] for l in pair.xblocks})))
                for l in [6,15,27]:
                    intervention_specs.append((f'clean_post_{x}_{l}',dict(post_donors={(x,l):traces['cut'].full_post[(x,l)]})))
                # Direction-specific content control for the largest late post-state patch.
                l=27;truth=traces['cut'].full_post[(x,l)];factual=traces['link'].full_post[(x,l)]
                displacement=truth.float()-factual.float()
                noise=link.randomize(displacement,torch.Generator().manual_seed(9021+bid))
                intervention_specs.append((f'random_post_{x}_{l}',dict(post_donors={(x,l):(factual.float()+noise).to(factual.dtype)})))
            if args.dissection and any(c['question_id']=='NCEOGA_2013_8_57' for c in chosen):
                x,y='quill','ember'
                norms={(x,l):traces['link'].full_write_norms[(x,l)] for l in pair.xblocks}
                donor={(y,l):traces['cut'].full_pre[(y,l)] for l in pair.xblocks}
                intervention_specs += [('norm_sham_all_quill',dict(write_norms=norms)),
                    ('clean_kv_all_ember',dict(sender_kv_donors=donor)),
                    ('clean_gate_all_ember',dict(sender_gate_donors=donor)),
                    ('clean_kv_normmatched_all_ember',dict(sender_kv_donors=donor,write_norms=norms))]
                for l in pair.xblocks:
                    donor={(y,l):traces['cut'].full_pre[(y,l)]}
                    intervention_specs += [(f'drop_reverse_ember_read_{l}',dict(drop_writes={(y,l)})),
                        (f'clean_kv_ember_{l}',dict(sender_kv_donors=donor)),
                        (f'clean_gate_ember_{l}',dict(sender_gate_donors=donor)),
                        (f'clean_kv_normmatched_ember_{l}',dict(sender_kv_donors=donor,write_norms={(x,l):norms[(x,l)]}))]
            for name,kw in intervention_specs:
                before=time.time()
                with CausalTrace(pair,positions,capture=False,**kw):
                    last,_=pair.joint(b)
                lg=letter_logits(pair,last,own);records[name]=outcomes(lg,items);timing[name]=time.time()-before
                if readouts:add_dot64_readout(records[name],last,own,readouts)
            for name,mode in [('shuffle','shuffle'),('random_sender','random'),('link_repeat',None)]:
                before=time.time();last,_=pair.joint(b,mode=mode,gen=torch.Generator().manual_seed(2021+bid))
                lg=letter_logits(pair,last,own);records[name]=outcomes(lg,items);timing[name]=time.time()-before
                if readouts:add_dot64_readout(records[name],last,own,readouts)
            checks={}
            for x in pair.models:
                y=link.partner(x);one='ember_to_quill_only' if x=='quill' else 'quill_to_ember_only'
                # Same-question clean sender K/V/gates should mediate exactly the one-way run.
                clean=torch.tensor([r['letter_logits'] for r in records['clean_sender_'+y][x]])
                one_way=torch.tensor([r['letter_logits'] for r in records[one][x]])
                torch.testing.assert_close(clean,one_way,rtol=0,atol=0)
                repeat=torch.tensor([r['letter_logits'] for r in records['link_repeat'][x]])
                factual=torch.tensor([r['letter_logits'] for r in records['link'][x]])
                checks[x]=dict(clean_sender_equals_direction_cut_bitwise=True,
                    max_abs_repeat_logit_error=float((repeat-factual).abs().max()))
                if 'norm_sham_all_quill' in records:
                    assert [r['letter_logits'] for r in records['norm_sham_all_quill'][x]]==[r['letter_logits'] for r in records['link'][x]],'norm-only sham changed the forward pass'
            metadata=[]
            for i,case in enumerate(chosen):
                x=case['target_receiver']
                stable=(all(records['cut'][s][i]['prediction']==case['saved_baseline']['cut'][s]['prediction'] for s in pair.models)
                    and records['link'][x][i]['prediction']==case['saved_baseline']['link'][x]['prediction'])
                metadata.append(dict(**case,new_batch_target_class_stable=stable,
                    batch_shape_logit_max_abs={c:{s:max(abs(a-bb) for a,bb in zip(records[c][s][i]['letter_logits'],case['saved_baseline'][c][s]['letter_logits'])) for s in pair.models} for c in ['link','cut']},
                    positions={s:positions[s][i].tolist() for s in positions},
                    ticks={s:b[s]['tick'][i,positions[s][i]].tolist() for s in positions}))
            save(path,dict(evidence_type='fresh_nl33w_causal_case_series',cases=metadata,conditions=records,
                tracing_stats={c:tr.stats for c,tr in traces.items()},checks=checks,timing_seconds=timing,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                shuffle_donors={x:[chosen[(i-shift)%len(chosen)]['question_id'] for i in range(len(chosen))] for x,shift in [('quill',1),('ember',-1)]},
                patch_note='Same-question full-position CUT sender/post states; original causal ticks/masks preserved. Norm-matched random post delta at layer27. All model/adapters identical across conditions.'))
            del traces
            save(out/'PROGRESS.json',dict(completed_cases=z,total_cases=len(cases),completed_batches=bid+1,
                elapsed_seconds=time.time()-start,updated_unix=time.time()))
            print(json.dumps(dict(completed_cases=z,elapsed_seconds=time.time()-start)),flush=True)
    save(out/'COMPLETE.json',dict(status='complete',n_cases=len(cases),gpu_worker_seconds=time.time()-start,finished_unix=time.time()))


if __name__=='__main__':main()
