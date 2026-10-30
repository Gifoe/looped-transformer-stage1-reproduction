"""Case-wise causal effects and layer diagnostics; no population/novelty claims."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
from linux_runtime import ROOT,save


def write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def interval(values):
    v=np.asarray(values,float)
    if not len(v):return [None,None]
    rng=np.random.default_rng(2021)
    b=v[rng.integers(0,len(v),size=(10000,len(v)))].mean(1)
    return np.quantile(b,[.025,.975]).tolist()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--partial',action='store_true');ap.add_argument('--study',default='causal_pilot_16');args=ap.parse_args()
    assert args.study.replace('_','').isalnum()
    out=ROOT/'fresh'/args.study
    if not args.partial:assert json.loads((out/'COMPLETE.json').read_text())['status']=='complete'
    rows=[];layers=[];cases=[];random_controls=[];checks=[]
    batches=sorted(out.glob('batch_*.json'))
    for path in batches:
        d=json.loads(path.read_text());bid=int(path.stem.split('_')[1]);conditions=d['conditions']
        checks.append(d['checks'])
        for i,case in enumerate(d['cases']):
            target=case['target_receiver'];other='ember' if target=='quill' else 'quill'
            stable=case['new_batch_target_class_stable']
            base=conditions['link'][target][i];cut=conditions['cut'][target][i]
            one='ember_to_quill_only' if target=='quill' else 'quill_to_ember_only'
            candidate=dict(question_id=case['question_id'],question_index=case['question_index'],receiver=target,
                category=case['category'],stable=stable,selected=case.get('selected',True),item=case['item'],link=base,cut=cut,
                partner_cut=conditions['cut'][other][i],partner_link=conditions['link'][other][i],
                one_way=conditions[one][target][i],clean_sender=conditions['clean_sender_'+other][target][i],
                clean_post27=conditions[f'clean_post_{target}_27'][target][i],
                random_post27=conditions[f'random_post_{target}_27'][target][i])
            candidate['best_drop_layer']=max([6,9,12,15,18,21,24,27],key=lambda l:conditions['drop_layer_'+str(l)][target][i]['correct_answer_margin'])
            candidate['best_drop']=conditions['drop_layer_'+str(candidate['best_drop_layer'])][target][i]
            cases.append(candidate)
            for name,records in conditions.items():
                for x in ['quill','ember']:
                    rec=records[x][i];on=conditions['link'][x][i];off=conditions['cut'][x][i]
                    last=name.rsplit('_',1)[-1]
                    row=dict(evidence_type='fresh_nl33w_causal_case_series',checkpoint='released_final_nl33w',
                        question_id=case['question_id'],question_index=case['question_index'],selected_category=case['category'],
                        target_receiver=target,receiver=x,is_target=x==target,batch_shape_class_stable=stable,
                        is_selected_case=case.get('selected',True),
                        intervention=name,layer=int(last) if last.isdigit() else '',gold=rec['gold'],prediction=rec['prediction'],correct=rec['correct'],
                        correct_answer_margin=rec['correct_answer_margin'],link_margin=on['correct_answer_margin'],cut_margin=off['correct_answer_margin'],
                        delta_margin_vs_link=rec['correct_answer_margin']-on['correct_answer_margin'],
                        delta_margin_vs_cut=rec['correct_answer_margin']-off['correct_answer_margin'],
                        repeat_logit_max_abs=d['checks'][x]['max_abs_repeat_logit_error'])
                    row.update({f'logit_{k}':v for k,v in zip('ABCD',rec['letter_logits'])})
                    row.update({f'probability_{k}':v for k,v in zip('ABCD',rec['letter_probabilities'])})
                    if 'dot64_logits' in rec:
                        row.update(dot64_prediction=rec['dot64_prediction'],dot64_correct=rec['dot64_correct'],
                            dot64_margin=rec['dot64_margin'],dot64_delta_margin_vs_link=rec['dot64_margin']-on['dot64_margin'],
                            bf16_dot64_argmax_disagree=rec['prediction']!=rec['dot64_prediction'])
                        row.update({f'dot64_logit_{k}':v for k,v in zip('ABCD',rec['dot64_logits'])})
                    rows.append(row)
        states={c:torch.load(out/f'batch_{bid:03d}_{c}_states.pt',map_location='cpu',weights_only=True) for c in ['link','cut']}
        for cond,stats in d['tracing_stats'].items():
            for s in stats:
                if s['selected_position']!=0:continue
                i=s['row'];x=s['receiver'];y='ember' if x=='quill' else 'quill';layer=s['layer'];case=d['cases'][i]
                st=states[cond];p=st[(x,layer,'attention')][i,:,0]
                ky=case['positions'][y][1]+1
                row=dict(evidence_type='fresh_nl33w_layer_diagnostic',question_id=case['question_id'],selected_category=case['category'],
                    condition=cond,receiver=x,is_target=x==case['target_receiver'],layer=layer,
                    write_hidden_ratio=s['write_hidden_ratio'],raw_write_l2=float(st[(x,layer,'raw_write_fp32')][i,0].norm()),
                    applied_write_l2=s['write_l2'],effective_bf16_write_l2=float(st[(x,layer,'effective_write')][i,0].norm()),
                    sender_last_visible_gate=float(st[(y,layer,'sender_gate')][i,1,0]),
                    sender_last_visible_k_l2=float(st[(y,layer,'shown_k')][i,1].norm()),
                    sender_last_visible_v_l2=float(st[(y,layer,'shown_v')][i,1].norm()),
                    null_attention_mass=s['null_attention_mass'],attention_entropy=float(-(p*p.clamp_min(1e-30).log()).sum(-1).mean()),
                    last_visible_partner_attention=float(p[:,ky].mean()),alpha=s['alpha'],
                    receiver_query_position=case['positions'][x][0],sender_last_visible_position=case['positions'][y][1])
                layers.append(row)
        # Reconstruct the exact CPU-seeded random donor perturbation to check the magnitude control.
        for x in ['quill','ember']:
            y='ember' if x=='quill' else 'quill';l=27
            n=len(d['cases']);length=states['link'][(y,l,'attention')].shape[-1]-1
            dim=states['link'][(x,l,'post')].shape[-1]
            noise=torch.randn((n,length,dim),generator=torch.Generator().manual_seed(9021+bid))
            for i,case in enumerate(d['cases']):
                position=case['positions'][x][0]
                base=states['link'][(x,l,'post')][i,0]
                clean=states['cut'][(x,l,'post')][i,0]
                delta=clean-base;r=F.normalize(noise[i,position],dim=-1)*delta.norm()
                actual=(base+r).to(torch.bfloat16).float()-base
                random_controls.append(dict(question_id=case['question_id'],receiver=x,is_target=x==case['target_receiver'],
                    clean_delta_l2=float(delta.norm()),random_effective_delta_l2=float(actual.norm()),
                    norm_ratio=float(actual.norm()/delta.norm().clamp_min(1e-12)),
                    cosine_clean_random=float(F.cosine_similarity(delta[None],actual[None])[0])))
    stable=[c for c in cases if c['stable']]
    summaries=[]
    for category in sorted({c['category'] for c in stable}):
        ids={c['question_id'] for c in stable if c['category']==category}
        for name in sorted({r['intervention'] for r in rows}):
            rr=[r for r in rows if r['is_target'] and r['question_id'] in ids and r['intervention']==name]
            if not rr:continue
            delta=[r['delta_margin_vs_link'] for r in rr]
            ci=interval(delta)
            summaries.append(dict(category=category,intervention=name,n=len(rr),correct=sum(r['correct'] for r in rr),
                mean_delta_margin=float(np.mean(delta)),median_delta_margin=float(np.median(delta)),
                descriptive_case_bootstrap_low=ci[0],descriptive_case_bootstrap_high=ci[1],
                scope='Selected case-series only; no benchmark-population or multiplicity-adjusted significance claim'))
    write(out/'intervention_results.csv',rows);write(out/'layerwise_analysis.csv',layers)
    write(out/'condition_summary.csv',summaries);write(out/'random_patch_control.csv',random_controls)
    save(out/'CASE_SUMMARIES.json',cases)
    report=dict(status='partial' if args.partial else 'complete',questions=len(cases),stable_questions=len(stable),
        selected_targets=sum(c['selected'] for c in cases),
        original_shapes=bool(json.loads((out/'SELECTION.json').read_text()).get('prescribed_plan')),
        unstable_ids=[c['question_id'] for c in cases if not c['stable']],condition_count=len({r['intervention'] for r in rows}),
        max_repeat_logit_error=max((s[x]['max_abs_repeat_logit_error'] for s in checks for x in s),default=0),
        mediation_identity_checks_all_passed=all(s[x]['clean_sender_equals_direction_cut_bitwise'] for s in checks for x in s),
        summaries=summaries,warnings=['Selection conditions on observed ON/CUT transitions; intervention recovery is not an estimate of general accuracy gains.',
            'Clean-sender equality with a direction cut is an implementation identity check, not independent replication.',
            'Probe-free case traces cannot uniquely establish information absence, misalignment or downstream utilization failure.',
            'No recurrence or architectural conclusion follows from a successful clean-state patch.'])
    save(out/'ANALYSIS.json',report)
    if not args.partial:
        write(ROOT/'intervention_results.csv',rows);write(ROOT/'layerwise_analysis.csv',layers)
    print(json.dumps({k:report[k] for k in ['status','questions','stable_questions','unstable_ids','condition_count','max_repeat_logit_error','mediation_identity_checks_all_passed']},indent=2))


if __name__=='__main__':main()
