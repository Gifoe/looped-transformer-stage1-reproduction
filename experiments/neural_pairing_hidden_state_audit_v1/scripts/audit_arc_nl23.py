"""Official nl23 ARC-only 83/50 strata from immutable correctness records."""
import json
from audit_published import ROOT,source,write_csv,paired

d,sha=source('results/magic-nl23/bench.json')
c=d['per_question'];ids=[i for i,q in enumerate(d['questions']) if q['src']=='arc_challenge']
assert len(ids)==1165
groups={'both_correct':lambda i:c['pair/zero/quill'][i] and c['pair/zero/ember'][i],
    'only_quill_correct':lambda i:c['pair/zero/quill'][i] and not c['pair/zero/ember'][i],
    'only_ember_correct':lambda i:not c['pair/zero/quill'][i] and c['pair/zero/ember'][i],
    'both_wrong':lambda i:not c['pair/zero/quill'][i] and not c['pair/zero/ember'][i]}
rows=[];selected={}
for name,predicate in groups.items():
    group=[i for i in ids if predicate(i)];selected[name]=group
    for side in ['quill','ember']:
        before=[bool(c['pair/zero/'+side][i]) for i in group];after=[bool(c['pair/'+side][i]) for i in group]
        a=sum(not b and t for b,t in zip(before,after));b=sum(t and not u for t,u in zip(before,after))
        right=sum(before);wrong=len(before)-right
        rows.append(dict(evidence_type='published_json_reanalysis',run='nl23',task='ARC-Challenge',exclusion='unfiltered_official_4choice',
            stratum=name,receiver=side,n=len(group),corrections=a,degradations=b,
            rescue_rate=a/wrong if wrong else None,damage_rate=b/right if right else None,
            specific_wrong_answer_adoption='not_identifiable_without_saved_letters',probability_change='unavailable'))
assert len(selected['only_quill_correct'])==83 and len(selected['only_ember_correct'])==50
write_csv(ROOT/'published_arc_nl23_disagreement.csv',rows)
(ROOT/'logs/PUBLISHED_ARC_NL23.json').write_text(json.dumps(dict(evidence_type='published_json_reanalysis',source_sha256=sha,
    n=1165,indices=selected,strata=rows,paired={x:paired([c['pair/'+x][i] for i in ids],[c['pair/zero/'+x][i] for i in ids]) for x in ['quill','ember']}),indent=2),encoding='utf8')
print(json.dumps(rows))
