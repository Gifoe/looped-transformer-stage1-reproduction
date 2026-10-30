"""Analyze completed fresh ARC ON/CUT predictions; refuse missing/duplicate questions."""
from pathlib import Path
import csv
import json
import numpy as np
from scipy.stats import beta,binomtest
from linux_runtime import ROOT,save


def paired(a,b):
    a,b=np.asarray(a,bool),np.asarray(b,bool);n=len(a)
    plus,minus=int((a&~b).sum()),int((~a&b).sum())
    lo=lambda k:float(beta.ppf(.0125,k,n-k+1)) if k else 0.
    hi=lambda k:float(beta.ppf(.9875,k+1,n-k)) if k<n else 1.
    boot=np.random.default_rng(2021).multinomial(n,[minus/n,(n-plus-minus)/n,plus/n],size=10000)
    ci=np.quantile((boot[:,2]-boot[:,0])/n*100,[.025,.975])
    return dict(n=n,corrections=plus,degradations=minus,net=plus-minus,delta_pp=(plus-minus)/n*100,
        conservative_ci_low_pp=(lo(plus)-hi(minus))*100,conservative_ci_high_pp=(hi(plus)-lo(minus))*100,
        paired_bootstrap_ci_low_pp=float(ci[0]),paired_bootstrap_ci_high_pp=float(ci[1]),
        exact_mcnemar_p=float(binomtest(plus,plus+minus).pvalue) if plus+minus else 1.)


def main():
    out=ROOT/'fresh/arc_first'
    done=json.loads((out/'COMPLETE.json').read_text());assert done['status']=='complete'
    n=done['n'];items=json.loads((out/'items.json').read_text())['items']
    data={c:{x:[None]*n for x in ['quill','ember']} for c in ['link','cut']}
    for cond in data:
        for p in sorted(out.glob('batch_*_'+cond+'.json')):
            rows=json.loads(p.read_text())
            for x in data[cond]:
                for i,r in zip(rows['question_indices'],rows['per_receiver'][x]):
                    assert data[cond][x][i] is None
                    data[cond][x][i]=r
        assert all(r is not None for records in data[cond].values() for r in records)
    baseline,comparisons,strata,allrows,error_cases=[],[],[],[],[]
    for cond in data:
        for x,records in data[cond].items():
            baseline.append(dict(evidence_type='fresh_final_checkpoint_inference',task='ARC-Challenge',checkpoint='final-nl33w',
                condition=cond,receiver=x,n=n,correct=sum(r['correct'] for r in records),accuracy=sum(r['correct'] for r in records)/n))
        probs=np.array([data[cond][x][i]['letter_probabilities'] for x in ['quill','ember'] for i in range(n)]).reshape(2,n,4).mean(0)
        predicted=probs.argmax(-1)
        baseline.append(dict(evidence_type='fresh_final_checkpoint_inference',task='ARC-Challenge',checkpoint='final-nl33w',
            condition=cond,receiver='probability_mean',n=n,correct=int(sum(p==it['answer'] for p,it in zip(predicted,items))),accuracy=float(np.mean([p==it['answer'] for p,it in zip(predicted,items)]))))
    cut={x:np.array([r['correct'] for r in data['cut'][x]],bool) for x in ['quill','ember']}
    groups=dict(both_correct=cut['quill']&cut['ember'],only_quill_correct=cut['quill']&~cut['ember'],
        only_ember_correct=~cut['quill']&cut['ember'],both_wrong=~cut['quill']&~cut['ember'])
    for x in cut:
        y='ember' if x=='quill' else 'quill'
        link=np.array([r['correct'] for r in data['link'][x]])
        comparisons.append(dict(receiver=x,contrast='link_minus_cut',**paired(link,cut[x])))
        for name,mask in groups.items():
            correct_to_wrong=int((mask&cut[x]&~link).sum());wrong_to_correct=int((mask&~cut[x]&link).sum())
            right=int((mask&cut[x]).sum());wrong=int((mask&~cut[x]).sum())
            damaged=[i for i in np.flatnonzero(mask&cut[x]&~link) if not cut[y][i]]
            copied=sum(data['link'][x][i]['prediction']==data['cut'][y][i]['prediction'] for i in damaged)
            damaged_on_wrong_partner=[i for i in np.flatnonzero(mask&cut[x]&~link) if not data['link'][y][i]['correct']]
            copied_on=sum(data['link'][x][i]['prediction']==data['link'][y][i]['prediction'] for i in damaged_on_wrong_partner)
            strata.append(dict(receiver=x,stratum=name,n=int(mask.sum()),corrections=wrong_to_correct,degradations=correct_to_wrong,
                rescue_rate=wrong_to_correct/wrong if wrong else None,damage_rate=correct_to_wrong/right if right else None,
                damaged_with_cut_wrong_partner=len(damaged),adopts_partner_cut_specific_wrong_letter=copied,
                adoption_rate=copied/len(damaged) if damaged else None,
                damaged_with_on_wrong_partner=len(damaged_on_wrong_partner),adopts_partner_on_specific_wrong_letter=copied_on,
                on_wrong_adoption_rate=copied_on/len(damaged_on_wrong_partner) if damaged_on_wrong_partner else None,
                interpretation='Descriptive adoption signature; does not identify causal over-deference'))
        for i in range(n):
            allrows.append(dict(question_index=i,receiver=x,gold=items[i]['answer'],
                cut_prediction=data['cut'][x][i]['prediction'],link_prediction=data['link'][x][i]['prediction'],
                cut_correct=bool(cut[x][i]),link_correct=bool(link[i]),
                cut_margin=data['cut'][x][i]['correct_answer_margin'],link_margin=data['link'][x][i]['correct_answer_margin'],
                delta_margin=data['link'][x][i]['correct_answer_margin']-data['cut'][x][i]['correct_answer_margin']))
            if cut[x][i] and not link[i]:
                error_cases.append(dict(receiver=x,question_index=i,item=items[i],
                    cut=data['cut'][x][i],link=data['link'][x][i],partner_cut=data['cut'][y][i],partner_link=data['link'][y][i],
                    delta_margin=allrows[-1]['delta_margin']))
    for name,rows in [('baseline_results.csv',baseline),('paired_comparisons.csv',comparisons),('disagreement_analysis.csv',strata),('per_question.csv',allrows)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    save(out/'ANALYSIS.json',dict(evidence_type='fresh_final_checkpoint_inference',baseline=baseline,paired=comparisons,strata=strata,
        n=n,raw_arc_n=1172,excluded_non_four_choice=7,
        warnings=['Final checkpoint was not QA trained; not an nl23 reproduction.',
                  'Full 1165 four-choice test, no author whole-benchmark overlap exclusion applied.',
                  'No training-seed uncertainty; no causal hidden-state mechanism identified by answer changes alone.']))
    save(out/'damage_cases.json',sorted(error_cases,key=lambda r:r['delta_margin']))
    print(json.dumps({'baseline':baseline,'paired':comparisons},indent=2))


if __name__=='__main__':main()
