"""Recover target spans from construction, checked against the historical template."""
from types import SimpleNamespace
from lm_eval.api.model import TemplateLM

def make_question(prefix,question,choices,reference_question_prompt):
    if not 2<=len(choices)<=8: raise ValueError('unsupported choice count')
    parts=[prefix,'Question: ']; spans={};cursor=len(prefix)+len('Question: ')
    def add_span(name,text):
        nonlocal cursor
        start=cursor;parts.append(text);cursor+=len(text);spans[name]=[start,cursor]
    add_span('question',question)
    for i,choice in enumerate(choices):
        parts.append('\n');cursor+=1
        add_span(f'label_{i}',chr(65+i)+'.')
        parts.append(' ');cursor+=1
        add_span(f'option_{i}',choice)
    parts.append('\n');cursor+=1
    add_span('answer_marker','Answer:')
    prompt=''.join(parts)
    if prompt!=prefix+reference_question_prompt(question,choices): raise AssertionError('historical template mismatch')
    return prompt,spans

def token_spans(tokenizer,context,continuation,char_spans):
    coder=SimpleNamespace(tok_encode=lambda text,**kw:tokenizer.encode(text,add_special_tokens=False))
    prefix,target=TemplateLM._encode_pair(coder,context,continuation)
    encoded=tokenizer(context,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids']!=prefix: raise AssertionError('context tokenization mismatch')
    if not prefix or not target: raise ValueError('empty context or continuation')
    mapping={}
    for name,(start,end) in char_spans.items():
        ids=[i for i,(a,b) in enumerate(encoded['offset_mapping']) if a<end and b>start and b>a]
        if not ids: raise ValueError('span has no tokens: '+name)
        mapping[name]=ids
    model_input=prefix+target[:-1]
    predecessor=list(range(len(prefix)-1,len(prefix)+len(target)-1))
    if max(predecessor)>=len(model_input): raise AssertionError('bad continuation alignment')
    return {'input_ids':model_input,'context_ids':prefix,'target_ids':target,'predecessor_positions':predecessor,'spans':mapping,'offset_mapping':encoded['offset_mapping'],'truncated':False}
