"""Structure-preserving rendering; query gold is never an input."""
ALPHABET='ABCDE'
def transform(block,shift,rename):
 assert set(rename)==set(ALPHABET) and set(rename.values())==set(ALPHABET)
 c=len(block['choices']);assert block['labels']==list(ALPHABET[:c]);s=shift%c;mapping=[(j+s)%c for j in range(c)];labels=[rename[l] for l in block['labels']]
 assert len(set(labels))==c and sorted(mapping)==list(range(c))
 answer=block['answer_text']
 if answer:
  old=answer.strip();assert old in block['labels'];original=block['labels'].index(old);position=mapping.index(original);answer=' '+labels[position]
 return {'stem':block['stem'],'labels':labels,'choices':[block['choices'][i] for i in mapping],'answer_text':answer,'display_to_original':mapping}
def render(block):
 return 'Question: '+block['stem']+''.join('\n'+l+'. '+s for l,s in zip(block['labels'],block['choices']))+'\nAnswer:'+block['answer_text']
def prompt(demonstrations,query,order_shift,rename):
 assert query['answer_text']==''
 # P acts only on the query. S is globally consistent across demonstrations/query.
 ds=[transform(q,0,rename) for q in demonstrations];qq=transform(query,order_shift,rename)
 return '\n\n'.join(render(b) for b in ds+[qq]),qq,ds
