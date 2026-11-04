"""Field-based reference rendering and independently encoded prefix alignment."""
import copy
import hashlib


def arc_references(row,render):
    doc=row['source_doc'];current=render(doc)
    if not current or not row['prompt'].endswith(current):raise ValueError('Current field-rendered document does not match prompt suffix')
    prefix=row['prompt'][:-len(current)]
    refs=['Answer:']
    for placeholder in ['N/A','Unknown']:
        reference=copy.deepcopy(doc);reference['question']=placeholder
        refs.append(prefix+render(reference))
    return refs,dict(fewshot_prefix_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
                     current_document_sha256=hashlib.sha256(current.encode()).hexdigest(),
                     current_field='question',operation='Structured field override and exact suffix validation; no string replacement')


def prefix_alignment(tokenizer,request,encoded,reference_pairs):
    """Align by consumed literal continuation TEXT, never by forced token IDs.

    Complete independently encoded target decoding must match the continuation.
    Prefixes splitting a Unicode code point or other non-round-tripping cases
    are explicitly unsupported and trigger the predeclared whole-question R6
    alignment fallback. Numerical/model/cache failures are not caught here.
    """
    _,continuation=request;_,target=encoded
    def prefixes(ids):
        decoded=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        if decoded!=continuation:return None
        values=[]
        for i in range(len(ids)):
            text=tokenizer.decode(ids[:i],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            if not continuation.startswith(text):return None
            values.append(text)
        if len(set(values))!=len(values):return None
        return values
    real=prefixes(target)
    if real is None:return None,'real_target_decode_or_prefix_boundary'
    mappings=[]
    for _,ref_target in reference_pairs:
        values=prefixes(ref_target)
        if values is None:return None,'reference_target_decode_or_prefix_boundary'
        index={text:i for i,text in enumerate(values)}
        if not all(text in index for text in real):return None,'no_reference_position_for_real_literal_prefix'
        mappings.append([index[text] for text in real])
    return mappings,None
