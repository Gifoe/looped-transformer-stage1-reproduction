"""Align author test indices, while preserving checkpoint-specific groups and metrics."""
import json
from audit_published import ROOT, source, write_csv


def stratum(q,e):
    return 'both_correct' if q and e else 'only_quill_correct' if q else 'only_ember_correct' if e else 'both_wrong'


def main():
    author,sha=source('results/magic-nl23/bench.json')
    c=author['per_question'];arc=[i for i,q in enumerate(author['questions']) if q['src']=='arc_challenge']
    items=json.loads((ROOT/'fresh/arc_first/items.json').read_text(encoding='utf8'))
    scores={k:{s:[None]*1165 for s in ['quill','ember']} for k in ['link','cut']}
    for k in scores:
        for p in (ROOT/'fresh/arc_first').glob('batch_*_'+k+'.json'):
            d=json.loads(p.read_text(encoding='utf8'))
            for s in scores[k]:
                for j,z in zip(d['question_indices'],d['per_receiver'][s]):scores[k][s][j]=z
    rows=[]
    assert len(arc)==len(items['items'])==1165
    for j,idx in enumerate(arc):
        q=author['questions'][idx];it=items['items'][j]
        assert q['answer']==it['answer'] and q['src']==it['src']
        row=dict(author_test_index=idx,arc_index=j,question_id=items['original_ids'][j],gold=it['answer'],
            nl23_evidence='published_correctness_only',nl33w_evidence='fresh_bf16_checkpoint_inference',
            nl23_cut_stratum=stratum(c['pair/zero/quill'][idx],c['pair/zero/ember'][idx]),
            nl33w_cut_stratum=stratum(scores['cut']['quill'][j]['correct'],scores['cut']['ember'][j]['correct']))
        for s in ['quill','ember']:
            row.update({f'nl23_{s}_cut_correct':c['pair/zero/'+s][idx],f'nl23_{s}_on_correct':c['pair/'+s][idx],
                f'nl33w_{s}_cut_correct':scores['cut'][s][j]['correct'],f'nl33w_{s}_on_correct':scores['link'][s][j]['correct']})
        rows.append(row)
    write_csv(ROOT/'checkpoint_separated_question_comparison.csv',rows)
    print(json.dumps(dict(n=len(rows),source_sha256=sha,alignment='Author ARC order; exact rebuilt bench SHA verified independently; src and gold checked per row.',
        warning='No pooling, transfer of strata, paired cross-checkpoint treatment effect, or claim of identical wrong answers.')))


if __name__=='__main__':main()
