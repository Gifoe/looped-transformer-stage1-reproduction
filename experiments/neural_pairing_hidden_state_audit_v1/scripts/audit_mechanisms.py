"""Describe original-shape causal records without treating selected cases as a trial."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from linux_runtime import ROOT, save


def table(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--study', default='causal_original_shape'); args = ap.parse_args()
    out = ROOT / 'fresh' / args.study
    assert json.loads((out / 'COMPLETE.json').read_text(encoding='utf8'))['status'] == 'complete'
    events, selected, precision, feedback = [], [], [], []
    base_mismatch = 0
    for p in sorted(out.glob('batch_*.json')):
        d = json.loads(p.read_text(encoding='utf8')); cond = d['conditions']
        for i, c in enumerate(d['cases']):
            for x in ['quill', 'ember']:
                y = 'ember' if x == 'quill' else 'quill'
                on, cut = cond['link'][x][i], cond['cut'][x][i]
                pc, po = cond['cut'][y][i], cond['link'][y][i]
                one_name = 'ember_to_quill_only' if x == 'quill' else 'quill_to_ember_only'
                one = cond[one_name][x][i]
                transition = 'harm' if cut['correct'] and not on['correct'] else 'rescue' if on['correct'] and not cut['correct'] else 'unchanged'
                base_mismatch += sum(c['batch_shape_logit_max_abs'][k][s] != 0 for k in ['cut', 'link'] for s in [x])
                row = dict(question_id=c['question_id'], receiver=x, selected=c['selected'] and c['target_receiver'] == x,
                    transition=transition, partner_cut_correct=pc['correct'], partner_on_correct=po['correct'],
                    gold=on['gold'], cut_prediction=cut['prediction'], link_prediction=on['prediction'],
                    partner_cut_prediction=pc['prediction'], partner_link_prediction=po['prediction'],
                    adopts_partner_cut_wrong=transition == 'harm' and not pc['correct'] and on['prediction'] == pc['prediction'],
                    adopts_partner_link_wrong=transition == 'harm' and not po['correct'] and on['prediction'] == po['prediction'],
                    cut_margin=cut['correct_answer_margin'], link_margin=on['correct_answer_margin'],
                    one_way_margin=one['correct_answer_margin'], one_way_correct=one['correct'],
                    dot64_cut_margin=cut['dot64_margin'], dot64_link_margin=on['dot64_margin'],
                    dot64_one_way_margin=one['dot64_margin'],
                    original_transition_survives_dot64=(cut['correct'] == cut['dot64_correct'] and on['correct'] == on['dot64_correct']),
                    reverse_cut_repairs_both_readouts=transition == 'harm' and one['correct'] and one['dot64_correct'] and not on['dot64_correct'],
                    drop_layers_repairing=';'.join(str(l) for l in [6,9,12,15,18,21,24,27] if transition == 'harm' and cond[f'drop_layer_{l}'][x][i]['correct']),
                    drop_layers_repairing_both_readouts=';'.join(str(l) for l in [6,9,12,15,18,21,24,27] if transition == 'harm' and cond[f'drop_layer_{l}'][x][i]['correct'] and cond[f'drop_layer_{l}'][x][i]['dot64_correct']),
                    half_write_correct=cond['write_scale_0.5'][x][i]['correct'],
                    random_sender_correct=cond['random_sender'][x][i]['correct'], shuffle_correct=cond['shuffle'][x][i]['correct'])
                events.append(row)
                if row['selected']: selected.append(dict(**row, question=c['item']['q'], choices=json.dumps(c['item']['choices'],ensure_ascii=False)))
                for name, records in cond.items():
                    z = records[x][i]
                    precision.append(dict(question_id=c['question_id'], receiver=x, condition=name,
                        bf16_prediction=z['prediction'], dot64_prediction=z['dot64_prediction'],
                        disagree=z['prediction'] != z['dot64_prediction'],
                        bf16_tie=sorted(z['letter_logits'])[-1] == sorted(z['letter_logits'])[-2],
                        max_logit_rounding_error=max(abs(a-b) for a,b in zip(z['letter_logits'],z['dot64_logits']))))
                if c['question_id'] == 'NCEOGA_2013_8_57':
                    for name, records in cond.items():
                        z = records[x][i]
                        feedback.append(dict(question_id=c['question_id'], receiver=x, condition=name,
                            correct=z['correct'], margin=z['correct_answer_margin'], dot64_correct=z['dot64_correct'],
                            dot64_margin=z['dot64_margin'], delta_vs_link=z['dot64_margin'] - on['dot64_margin']))
    table(out/'causal_events.csv', events); table(out/'selected_case_table.csv', selected)
    table(out/'readout_precision.csv', precision); table(out/'feedback_dissection.csv', feedback)
    groups={}
    for x in ['quill','ember','both_receivers']:
        rr=[r for r in events if x=='both_receivers' or r['receiver']==x]
        harms=[r for r in rr if r['transition']=='harm']; gains=[r for r in rr if r['transition']=='rescue']
        groups[x]=dict(n_receiver_events=len(rr),harm=len(harms),rescue=len(gains),
            harm_with_cut_correct_partner=sum(r['partner_cut_correct'] for r in harms),
            harm_with_both_cut_correct_and_on_correct_partner=sum(r['partner_cut_correct'] and r['partner_on_correct'] for r in harms),
            harms_surviving_dot64=sum(r['original_transition_survives_dot64'] for r in harms),
            reverse_cut_repairs_both_readouts=sum(r['reverse_cut_repairs_both_readouts'] for r in harms),
            feedback_correct_peer_candidates=[r['question_id'] for r in harms if r['reverse_cut_repairs_both_readouts'] and r['partner_cut_correct'] and r['partner_on_correct']],
            forward_damage_survives_oneway_both_readouts=sum(r['original_transition_survives_dot64'] and not r['one_way_correct'] and r['dot64_one_way_margin']<0 for r in harms),
            rescued_both_cut_wrong=sum(not r['partner_cut_correct'] for r in gains),
            gains_retained_oneway_both_readouts=sum(r['one_way_correct'] and r['dot64_one_way_margin']>0 for r in gains),
            harms_adopt_partner_cut_wrong=sum(r['adopts_partner_cut_wrong'] for r in harms),
            harms_adopt_partner_link_wrong=sum(r['adopts_partner_link_wrong'] for r in harms))
    gate = dict(original_baseline_logit_mismatches=base_mismatch,
        same_shape_repeats_max_abs=max(d['checks'][x]['max_abs_repeat_logit_error'] for p in out.glob('batch_*.json') for d in [json.loads(p.read_text())] for x in ['quill','ember']),
        precision_diagnostic_records=len(precision),precision_argmax_disagreements=sum(r['disagree'] for r in precision),
        original_on_cut_argmax_disagreements=sum(r['disagree'] for r in precision if r['condition'] in ['link','cut']))
    save(out/'MECHANISM_AUDIT.json',dict(scope='Six selected original batches,95 questions,190 receiver events; case-series, not population prevalence.',
        groups=groups,checks=gate,selected_cases=selected,
        intervention_warning='All-layer clean-sender mediation and last-bridge clean-post restoration are identities, not discoveries. A successful patch need not reveal semantic content.'))
    print(json.dumps(dict(groups=groups,checks=gate),indent=2))


if __name__ == '__main__': main()
