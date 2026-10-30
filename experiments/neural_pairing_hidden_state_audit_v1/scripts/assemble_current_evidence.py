"""Merge completed evidence types without modifying immutable source records."""
import csv
import json
from linux_runtime import ROOT


def read(path):
    with path.open(newline='',encoding='utf8') as f:return list(csv.DictReader(f))


def write(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main():
    original=read(ROOT/'baseline_results.csv')
    original=[r for r in original if r['evidence_type']=='published_json_reanalysis']
    a=json.loads((ROOT/'fresh/arc_first/ANALYSIS.json').read_text())
    for r in a['baseline']:
        side=r['receiver'];arm=('pair/' if r['condition']=='link' else 'pair/zero/')+side
        original.append(dict(evidence_type='fresh_final_checkpoint_inference',run='released_nl33w',
            exclusion='official_four_choice_unfiltered',task='arc_challenge',arm=arm,n=r['n'],correct=r['correct'],
            accuracy=r['accuracy'],upstream_commit='7e7fcc401a8243f94bca94070db0993d64277874'))
    write(ROOT/'baseline_results.csv',original)
    rows=[r for r in read(ROOT/'disagreement_analysis.csv') if r['evidence_type']=='published_json_reanalysis' and r.get('task','all')=='all']
    for r in rows:r.update(task='all',exclusion='exact_input_overlap')
    rows+=read(ROOT/'published_arc_nl23_disagreement.csv')
    n=a['n'];prob={c:{x:[None]*n for x in ['quill','ember']} for c in ['link','cut']}
    for cond in prob:
        for p in (ROOT/'fresh/arc_first').glob('batch_*_'+cond+'.json'):
            d=json.loads(p.read_text())
            for x in prob[cond]:
                for i,r in zip(d['question_indices'],d['per_receiver'][x]):prob[cond][x][i]=r
    q=[r['correct'] for r in prob['cut']['quill']];e=[r['correct'] for r in prob['cut']['ember']]
    groups={'both_correct':[i for i in range(n) if q[i] and e[i]],'only_quill_correct':[i for i in range(n) if q[i] and not e[i]],
        'only_ember_correct':[i for i in range(n) if e[i] and not q[i]],'both_wrong':[i for i in range(n) if not q[i] and not e[i]]}
    for r in a['strata']:
        ids=groups[r['stratum']];x=r['receiver']
        dp=sum(prob['link'][x][i]['letter_probabilities'][prob['link'][x][i]['gold']]-prob['cut'][x][i]['letter_probabilities'][prob['cut'][x][i]['gold']] for i in ids)/len(ids)
        dm=sum(prob['link'][x][i]['correct_answer_margin']-prob['cut'][x][i]['correct_answer_margin'] for i in ids)/len(ids)
        rows.append(dict(evidence_type='fresh_final_checkpoint_inference',run='released_nl33w',task='arc_challenge',
            exclusion='official_four_choice_unfiltered',**r,probability_change=dp,margin_change=dm))
    write(ROOT/'disagreement_analysis.csv',rows)
    print(json.dumps(dict(baseline_rows=len(original),disagreement_rows=len(rows),
        types=['published_json_reanalysis','fresh_final_checkpoint_inference'],strict_nl23_inference_complete=False)))


if __name__=='__main__':main()
