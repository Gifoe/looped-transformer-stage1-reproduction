"""Independent CPU reconstruction of B prompts, tokens, mappings and cost.

Reads historical training questions, constructor metadata, and the inference
graph. Does not read B model outcomes, run a model, or modify A/R11.
"""
from pathlib import Path
import collections
import hashlib
import itertools
import json
import sys

from tokenizers import Tokenizer

P=Path(__file__).resolve().parent
ROOT=P.parent
H=lambda b:hashlib.sha256(b).hexdigest()
D=lambda v:H(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
read=lambda name:json.loads((P/name).read_text(encoding='utf8'))
plan=read('INFERENCE_TASK_PLAN.json')
constructor=read('CONSTRUCTOR_WITH_LABELS.json')
audit=read('prompt_and_target_audit.json')
protocol=read('intervention_protocol.json')
selection=read('ROOT_SHA_COHORT.json')
request_texts=read('INFERENCE_REQUEST_TEXTS.json')
assert H((P/'INFERENCE_REQUEST_TEXTS.json').read_bytes())==plan['request_texts_sha256']
tokenpath=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'artifacts/r24_TOKENIZER_AUDIT_PRIVATE.json'
assert H(tokenpath.read_bytes())=='fcb808fe5e7642f5299be28aea07fc7f6d4f4364c3ac5e408e15a772cbc8fa8d'
tok=Tokenizer.from_file(str(tokenpath))

def question(q,choices):
    return 'Question: '+q+'\n'+'\n'.join(chr(65+i)+'. '+s for i,s in enumerate(choices))+'\nAnswer:'

demos=json.loads((ROOT/'data/slcd_fixed_demonstrations.json').read_text(encoding='utf8'))
assert len(demos['documents'])==len(demos['ids'])==25
assert constructor['demo_ids']==demos['ids'] and constructor['demo_order_unchanged']
letter=[]
full=[]
for doc,identifier in zip(demos['documents'],demos['ids']):
    assert doc['id']==identifier
    ix=doc['choices']['label'].index(doc['answerKey'])
    prompt=question(doc['question'],doc['choices']['text'])
    letter.append(prompt+' '+chr(65+ix))
    full.append(prompt+' '+doc['choices']['text'][ix])
prefix={'letter':'\n\n'.join(letter)+'\n\n','full':'\n\n'.join(full)+'\n\n'}
assert H(prefix['letter'].encode())==constructor['letter_demo_prefix_sha256']
assert H(prefix['full'].encode())==constructor['fullanswer_demo_prefix_sha256']
source={}
for split in ('discovery','selection'):
    source.update((r['id'],r) for r in (json.loads(s) for s in (ROOT/f'data/slcd_{split}.jsonl').read_text(encoding='utf8').splitlines()))
assert [q['id'] for q in constructor['cohort']]==selection['pilot24_ids']
assert all('/train/' in x for x in selection['pilot24_ids'])
groups={(g['id'],g['condition']):g for g in constructor['groups']}
assert len(groups)==196
for rank,record in enumerate(constructor['cohort']):
    row=source[record['id']]
    assert record['question']==row['question'] and record['choices']==row['choices'] and record['gold']==row['gold']
    n=len(row['choices']);gold=row['gold']
    j=min((i for i in range(n) if i!=gold),key=lambda i:(H(('2021|'+row['id']+'|swap|'+str(i)).encode()),i))
    assert record['wrong_slot']==j
    identity=list(range(n));cycle=[(i+1)%n for i in range(n)]
    swap=identity.copy();swap[gold],swap[j]=swap[j],swap[gold]
    cyclic={tuple((i+s)%n for i in range(n)) for s in range(n)}
    noncyclic=next(list(p) for p in itertools.permutations(range(n)) if p not in cyclic and list(p)!=swap)
    for condition,order in [('P0',identity),('P1',cycle),('P2',swap),('P3',noncyclic)]:
        g=groups[(row['id'],condition)]
        assert g['question_text']==row['question'] and g['choices']==[row['choices'][i] for i in order] and g['order_to_original']==order
        original=g.get('expected_original_content_index',g.get('expected_index'))
        assert original==gold
        if 'expected_display_index' in g:assert g['expected_display_index']==order.index(gold)
    for condition in ('Q_placeholder','N_original','N_placeholder'):
        g=groups[(row['id'],condition)];expected=list(row['choices'])
        if condition.endswith('placeholder'):expected[j]='Not a valid answer'
        assert g['choices']==expected and g['choices'][gold]==row['choices'][gold]
        assert g['question_text']==('N/A' if condition.startswith('N_') else row['question'])
    if rank<4:
        for condition in ('Q_matched','N_matched'):
            g=groups[(row['id'],condition)];assert g['choices'][j]==record['secondary']
            assert all(g['choices'][i]==row['choices'][i] for i in range(n) if i!=j)
        for condition,demo,target in [('B5_L_full','letter','full'),('B5_F_letter','full','letter'),('B5_F_full','full','full')]:
            g=groups[(row['id'],condition)];assert g['choices']==row['choices'] and g['question_text']==row['question']
            assert g['demo_format']==demo and g['target_format']==target
    else:
        assert not any((row['id'],c) in groups for c in ('Q_matched','N_matched','B5_L_full','B5_F_letter','B5_F_full','B6_copy','B6_lookup'))

taskmap={x['task_key']:x for x in plan['tasks']}
assert len(taskmap)==len(plan['tasks'])==218
consumers={c['consumer_id']:(t,c) for t in plan['tasks'] for c in t['consumers']}
assert len(consumers)==784
audits={x['consumer_id']:x for x in audit['records']}
assert set(consumers)==set(audits)
for g in constructor['groups']:
    context=prefix[g['demo_format']]+question(g['question_text'],g['choices'])
    assert context==g['prompt_text'] and H(context.encode())==g['prompt_sha256']
    for info in g['consumers']:
        i=info['display_index'];t,c=consumers[info['consumer_id']];a=audits[info['consumer_id']]
        target=' '+(chr(65+i) if g['target_format']=='letter' else g['choices'][i])
        assert request_texts[info['consumer_id']]==dict(context=context,continuation=target)
        # All exact constructed contexts end in colon, so trailing-space
        # relocation in lm-eval's encode_pair is a verified no-op here.
        assert context==context.rstrip()
        pre=tok.encode(context,add_special_tokens=False).ids
        tar=tok.encode(context+target,add_special_tokens=False).ids[len(pre):]
        ids=pre+tar[:-1];pos=list(range(len(pre)-1,len(pre)+len(tar)-1))
        assert ids==t['input_ids'] and pos==t['positions'] and tar==c['target_ids']
        assert a['context_sha256']==H(context.encode()) and a['request_sha256']==D((context,target))
        assert a['target_ids_sha256']==D(tar) and a['input_ids_sha256']==D(ids) and a['prefix_ids_sha256']==D(pre)
        denominator=len(g['choices'][i]) if g['target_format']=='full' else 1
        assert c['char_denominator']==a['char_denominator']==denominator>0
        assert t['task_key']==D(dict(ids=ids,positions=pos,depth=4,head_shape=[1,len(tar),2048],primitive='original_uncached_no_explicit_mask_BF16_native_depth4',model_revision='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'))

def forbidden(obj):
    if isinstance(obj,dict):
        assert not any(k in ('gold','answerKey','expected_index','expected_display_index','question_text','choices','condition') for k in obj)
        for value in obj.values():forbidden(value)
    elif isinstance(obj,list):
        for value in obj:forbidden(value)
forbidden(plan)
forbidden(request_texts)
for q in constructor['cohort'][:4]:
    for original,modified in [('P0','Q_matched'),('N_original','N_matched')]:
        a=groups[(q['id'],original)];b=groups[(q['id'],modified)];j=q['wrong_slot']
        ea=tok.encode(a['prompt_text'],add_special_tokens=False);eb=tok.encode(b['prompt_text'],add_special_tokens=False)
        spans=[]
        for g,e in ((a,ea),(b,eb)):
            f=g['source_fields']['options'][j]
            spans.append([k for k,(x,y) in enumerate(e.offsets) if x<y and x<f['end'] and y>f['start']])
        sa,sb=spans;assert sa==sb and sa and len(ea.ids)==len(eb.ids)
        assert ea.ids[:sa[0]]==eb.ids[:sb[0]] and ea.ids[sa[-1]+1:]==eb.ids[sb[-1]+1:]

tasks=plan['tasks'];squared=sum(len(t['input_ids'])**2 for t in tasks)
positions=4*sum(len(t['positions']) for t in tasks)
timing=json.loads((ROOT/'recurrent_decision_audit_v1/a4_TIMINGS.json').read_text(encoding='utf8'))
warm=[x for x in timing if x['purpose']=='parity_native4'][1:]
rate=max(x['seconds']/x['length']**2 for x in warm)
cost=1.5*rate*squared+120+20+.02*4*sum(max(0,len(t['positions'])-1) for t in tasks)
assert squared==792321969 and positions==1448
assert abs(cost-protocol['cost']['forecast_total_seconds'])<1e-10 and cost<680<710.6733033489436
result=dict(status='passed_Root_independent_input_graph_reconstruction',questions=24,groups=196,
    all_784_complete_targets_prompt_hashes_tokens_and_mappings_exact=True,model_tasks=218,
    native_head_calls=872,native_head_prediction_positions=positions,
    all_eight_matched_prompt_prefix_suffix_ID_positions_exact=True,
    inference_plan_has_no_gold_labels_or_semantic_condition_names=True,
    all25_demo_ids_order_and_outputs_independently_reconstructed=True,
    planning_GPU_seconds=cost,proposed_hard_seconds=680,
    secondary_coverage=4,secondary_scope_not_attempted=20,
    no_model_or_intervention_outcomes_read=True,
    files_sha256={n:H((P/n).read_bytes()) for n in ('INFERENCE_TASK_PLAN.json','INFERENCE_REQUEST_TEXTS.json','CONSTRUCTOR_WITH_LABELS.json','prompt_and_target_audit.json','intervention_protocol.json')})
(P/'ROOT_INPUT_GRAPH_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
