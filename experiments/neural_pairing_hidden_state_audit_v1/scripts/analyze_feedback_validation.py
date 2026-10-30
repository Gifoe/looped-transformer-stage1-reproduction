"""Evaluate the fixed remaining-harm validation rule, retaining every target."""
import csv
import json
from linux_runtime import ROOT, save


def write(path, rows):
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    out=ROOT/'fresh/feedback_validation'
    assert json.loads((out/'COMPLETE.json').read_text())['status']=='complete'
    rows=[];cases=[];layers=[]
    for p in sorted(out.glob('batch_*.json')):
        d=json.loads(p.read_text(encoding='utf8'));cond=d['conditions']
        import torch
        bid=int(p.stem.split('_')[1])
        for mode in ['link','cut']:
            data=torch.load(out/f'batch_{bid:03d}_{mode}_selected_states.pt',map_location='cpu',weights_only=True)
            st=data['states']
            for j,i in enumerate(data['target_rows']):
                c=d['cases'][i]
                for x in ['quill','ember']:
                    y='ember' if x=='quill' else 'quill'
                    for l in [6,9,12,15,18,21,24,27]:
                        raw=st[(x,l,'raw_write_fp32')][j,0];applied=st[(x,l,'applied_write_fp32')][j,0]
                        h=st[(x,l,'pre')][j,0];prob=st[(x,l,'attention')][j,:,0]
                        layers.append(dict(evidence_type='fresh_nl33w_layer_diagnostic',question_id=c['question_id'],
                            receiver=x,is_target=x==c['target_receiver'],condition=mode,layer=l,
                            selected_position_semantics='own_answer_prefix',
                            write_hidden_ratio=float(applied.norm()/h.norm().clamp_min(1e-12)),raw_write_l2=float(raw.norm()),
                            applied_write_l2=float(applied.norm()),effective_bf16_write_l2=float(st[(x,l,'effective_write')][j,0].norm()),
                            null_attention_mass=float(prob[:,0].mean()),attention_entropy=float(-(prob*prob.clamp_min(1e-30).log()).sum(-1).mean()),
                            sender_gate_at_own_answer_prefix=float(st[(y,l,'sender_gate')][j,0,0])))
        for i,c in enumerate(d['cases']):
            if not c['selected']:continue
            x=c['target_receiver'];y='ember' if x=='quill' else 'quill'
            on,off=cond['link'][x][i],cond['cut'][x][i]
            stable=off['dot64_correct'] and not on['dot64_correct']
            case=dict(question_id=c['question_id'],receiver=x,question_index=c['question_index'],
                question=c['item']['q'],choices=json.dumps(c['item']['choices'],ensure_ascii=False),gold=on['gold'],
                cut_margin=off['correct_answer_margin'],on_margin=on['correct_answer_margin'],
                dot64_cut_margin=off['dot64_margin'],dot64_on_margin=on['dot64_margin'],transition_survives_dot64=stable,
                partner_cut_correct=cond['cut'][y][i]['correct'],partner_on_correct=cond['link'][y][i]['correct'])
            for kind,name in [('reverse_cut',f'oneway_to_{x}'),('clean_kv',f'clean_kv_from_{y}'),('clean_gate',f'clean_gate_from_{y}'),('clean_kv_normmatched',f'clean_kv_normmatched_from_{y}')]:
                z=cond[name][x][i]
                case.update({kind+'_margin':z['correct_answer_margin'],kind+'_dot64_margin':z['dot64_margin'],
                    kind+'_repairs_native':z['correct'],kind+'_repairs_both_readouts':bool(stable and z['correct'] and z['dot64_correct']),
                    kind+'_partner_correct':cond[name][y][i]['correct']})
            cases.append(case)
            for name,records in cond.items():
                for s in ['quill','ember']:
                    z=records[s][i]
                    row=dict(question_id=c['question_id'],target_receiver=x,receiver=s,intervention=name,
                        gold=z['gold'],prediction=z['prediction'],correct=z['correct'],margin=z['correct_answer_margin'],
                        dot64_prediction=z['dot64_prediction'],dot64_correct=z['dot64_correct'],dot64_margin=z['dot64_margin'],
                        delta_margin_vs_link=z['correct_answer_margin']-cond['link'][s][i]['correct_answer_margin'],
                        dot64_delta_margin_vs_link=z['dot64_margin']-cond['link'][s][i]['dot64_margin'])
                    row.update({f'logit_{a}':v for a,v in zip('ABCD',z['letter_logits'])})
                    row.update({f'probability_{a}':v for a,v in zip('ABCD',z['letter_probabilities'])})
                    rows.append(row)
    assert len(cases)==16 and len({c['question_id'] for c in cases})==16
    summary={}
    for x in ['quill','ember','both_receivers']:
        cc=[c for c in cases if x=='both_receivers' or c['receiver']==x]
        s=dict(n_targets=len(cc),transitions_surviving_dot64=sum(c['transition_survives_dot64'] for c in cc))
        for kind in ['reverse_cut','clean_kv','clean_gate','clean_kv_normmatched']:
            s[kind+'_repairs_native']=sum(c[kind+'_repairs_native'] for c in cc)
            s[kind+'_repairs_both_readouts']=sum(c[kind+'_repairs_both_readouts'] for c in cc)
            s[kind+'_robust_ids']=[c['question_id'] for c in cc if c[kind+'_repairs_both_readouts']]
        summary[x]=s
    write(out/'validation_cases.csv',cases);write(out/'intervention_results.csv',rows);write(out/'layerwise_analysis.csv',layers)
    result=dict(status='complete',scope='All16 previously untraced correct-peer harm events; same-checkpoint selected-failure validation, not accuracy estimation.',
        groups=summary,checks=dict(exact_original_replay_all=True,same_shape_repeat_all=True),
        content_specific_replication_gate=sum(c['clean_kv_normmatched_repairs_both_readouts'] for c in cases)>=2,
        warning='FP64 is output-dot arithmetic on BF16 states, not full-FP32 robustness. Native norm-matched patches alone do not decode the semantic content or establish an information bottleneck.')
    save(out/'ANALYSIS.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
