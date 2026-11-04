"""Structured R20 spans; CPU-only tokenizer preflight, no model imports."""
import hashlib
import numpy as np

def spans(row,tok,encode_pair,demonstrations):
    q=row['source_doc']['question'];current='Question: '+q+'\nAnswer:';prompt=row['prompt']
    assert prompt.endswith(current)
    few=prompt[:-len(current)];assert few+current==prompt
    # Verify each of the fixed 25 complete demonstration renders, no text search.
    assert len(demonstrations)==25
    rendered=''.join('Question: '+d['question']+'\nAnswer: '+d['answer']+'\n\n' for d in demonstrations)
    assert rendered==few
    a=len(few)+len('Question: ');b=a+len(q);assert b>a
    demo_bounds=[];pos=0
    for d in demonstrations:
        end=pos+len('Question: '+d['question']+'\nAnswer: '+d['answer']+'\n\n');demo_bounds.append([pos,end]);pos=end
    all_prefix=[];offsets=None
    for context,answer in row['request_pairs']:
        assert context==prompt
        prefix,target=encode_pair(tok,context,answer)
        encoded=tok(context,add_special_tokens=False,return_offsets_mapping=True)
        assert encoded['input_ids']==prefix,'Offset IDs differ from actual encode_pair prefix'
        assert target
        all_prefix.append(prefix)
        if offsets is None:offsets=encoded['offset_mapping']
        else:assert offsets==encoded['offset_mapping']
    assert all(x==all_prefix[0] for x in all_prefix)
    selected=[i for i,(s,e) in enumerate(offsets) if e>s and s<b and e>a]
    assert selected and selected==list(range(selected[0],selected[-1]+1))
    legal=[i for i in range(len(offsets)-len(selected)+1) if all(0<=s<e<=len(few) for s,e in offsets[i:i+len(selected)])]
    assert legal,'No legal demo window'
    seed=int.from_bytes(hashlib.sha256(('2021:'+row['id']).encode()).digest()[:8],'little')
    start=legal[int(np.random.default_rng(seed).integers(len(legal)))];demo=list(range(start,start+len(selected)))
    boundary=[dict(position=i,offset=list(offsets[i])) for i in selected if offsets[i][0]<a or offsets[i][1]>b]
    return dict(id=row['id'],question_char_interval=[a,b],fewshot_char_interval=[0,len(few)],demo_char_intervals=demo_bounds,question_positions=selected,demo_positions=demo,legal_demo_window_count=len(legal),demo_seed=seed,boundary_tokens=boundary,prefix_ids_sha256=hashlib.sha256(str(all_prefix[0]).encode()).hexdigest(),prefix_before_question=few+'Question: ',question=q,suffix='\nAnswer:',all_candidates_share_prefix=True)
