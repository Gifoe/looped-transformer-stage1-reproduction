"""Full1165 answer-letter transitions from fresh nl33w records only."""
import json
from audit_published import ROOT, write_csv


def main():
    out=ROOT/'fresh/arc_first';items=json.loads((out/'items.json').read_text(encoding='utf8'))
    assert json.loads((out/'COMPLETE.json').read_text())['status']=='complete'
    scores={k:{s:[None]*1165 for s in ['quill','ember']} for k in ['link','cut']}
    for k in scores:
        for p in out.glob('batch_*_'+k+'.json'):
            d=json.loads(p.read_text(encoding='utf8'))
            for s in scores[k]:
                for i,z in zip(d['question_indices'],d['per_receiver'][s]):scores[k][s][i]=z
    rows=[];summary={}
    for x,y in [('quill','ember'),('ember','quill')]:
        rr=[]
        for i,it in enumerate(items['items']):
            cut,on=scores['cut'][x][i],scores['link'][x][i];pc,po=scores['cut'][y][i],scores['link'][y][i]
            transition='C→W' if cut['correct'] and not on['correct'] else 'W→C' if on['correct'] and not cut['correct'] else 'C→C' if on['correct'] else 'W→W'
            row=dict(checkpoint='released_final_nl33w',question_id=items['original_ids'][i],arc_index=i,receiver=x,
                gold=it['answer'],transition=transition,cut_prediction=cut['prediction'],on_prediction=on['prediction'],
                partner_cut_prediction=pc['prediction'],partner_on_prediction=po['prediction'],partner_cut_correct=pc['correct'],partner_on_correct=po['correct'],
                cut_disagreement=cut['prediction']!=pc['prediction'],cut_margin=cut['correct_answer_margin'],on_margin=on['correct_answer_margin'],
                delta_margin=on['correct_answer_margin']-cut['correct_answer_margin'],
                gold_probability_cut=cut['letter_probabilities'][it['answer']],gold_probability_on=on['letter_probabilities'][it['answer']],
                adopts_cut_wrong_partner_answer=transition=='C→W' and not pc['correct'] and on['prediction']==pc['prediction'],
                adopts_on_wrong_partner_answer=transition=='C→W' and not po['correct'] and on['prediction']==po['prediction'])
            rr.append(row)
        rows+=rr
        harms=[r for r in rr if r['transition']=='C→W'];gains=[r for r in rr if r['transition']=='W→C']
        summary[x]=dict(n=1165,harm=len(harms),rescue=len(gains),
            harms_when_partner_cut_correct=sum(r['partner_cut_correct'] for r in harms),
            harms_when_partner_cut_and_on_correct=sum(r['partner_cut_correct'] and r['partner_on_correct'] for r in harms),
            harms_matching_wrong_partner_cut_answer=sum(r['adopts_cut_wrong_partner_answer'] for r in harms),
            harms_matching_wrong_partner_on_answer=sum(r['adopts_on_wrong_partner_answer'] for r in harms),
            rescues_both_cut_wrong=sum(not r['partner_cut_correct'] for r in gains),
            rescues_while_partner_on_wrong=sum(not r['partner_on_correct'] for r in gains))
    write_csv(out/'answer_transition_audit.csv',rows)
    (out/'TRANSITION_AUDIT.json').write_text(json.dumps(dict(checkpoint='released_final_nl33w',summary=summary,
        warning='Matching the partner wrong letter is association, not proof of copying; the receiver cannot see its future gold-answer token.'),indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
