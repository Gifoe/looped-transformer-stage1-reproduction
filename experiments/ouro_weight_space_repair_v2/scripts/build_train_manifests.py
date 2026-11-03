"""Freeze training-only cohorts and demonstrations; never opens validation/test."""
from pathlib import Path
import hashlib,json,random
import pyarrow.parquet as pq
from datasets import Dataset,DatasetDict
from lm_eval.api.task import ConfigurableTask
from runtime import ROOT,save,configure

configure()
def fingerprint(text):return hashlib.sha256(' '.join(text.casefold().split()).encode()).hexdigest()
def record(doc,index):
 return dict(id='arc_challenge/train/'+doc['id'],source_index=index,question=doc['question'],choices=doc['choices']['text'],
   gold=doc['choices']['label'].index(str(doc['answerKey'])),source_doc=doc)
def engine(demos,docs):
 data=DatasetDict(train=Dataset.from_list(demos),research=Dataset.from_list(docs))
 config=dict(task='arc_challenge',dataset_path='allenai/ai2_arc',dataset_name='ARC-Challenge',
   custom_dataset=lambda **kw:data,dataset_kwargs={},output_type='multiple_choice',training_split='train',
   validation_split=None,test_split='research',fewshot_split='train',num_fewshot=25,
   fewshot_config={'sampler':'first_n'},doc_to_text='Question: {{question}}\nAnswer:',
   doc_to_target='{{choices.label.index(answerKey)}}',doc_to_choice='{{choices.text}}',
   metric_list=[{'metric':'acc_norm','aggregation':'mean','higher_is_better':True}])
 out=ConfigurableTask(config=config);out.set_fewshot_seed(2021);return out
def render(rows,demos):
 out=[];e=engine(demos,[x['source_doc'] for x in rows])
 for row in rows:
  doc=row['source_doc'];context=e.fewshot_context(doc,25)
  instances=e.construct_requests(doc,context,metadata=('arc_challenge',row['source_index'],1))
  pairs=[list(x.args) for x in instances];choices=e.doc_to_choice(doc)
  assert all(x[0]==context for x in pairs) and int(e.doc_to_target(doc))==row['gold']
  out.append(dict(**row,prompt=context,prompt_sha256=hashlib.sha256(context.encode()).hexdigest(),request_pairs=pairs,
    choice_char_lengths=[len(c) for c in choices],fewshot_count=25,primary_metric='acc_norm'))
 return out
def write(name,rows):
 p=ROOT/'data'/name;payload=''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows)
 if p.exists():assert p.read_text(encoding='utf8')==payload,'Frozen manifest would change'
 p.write_text(payload,encoding='utf8');return dict(n=len(rows),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
 raw=pq.read_table(ROOT/'data/raw/train.parquet').to_pylist();assert len(raw)==1119
 indices=random.Random(2021).sample(range(len(raw)),25);demos=[raw[i] for i in indices]
 blocked={fingerprint(x['question']) for x in demos};seen=set(blocked);pool=[]
 for i,doc in enumerate(raw):
  key=fingerprint(doc['question'])
  if key in seen:continue
  seen.add(key);pool.append(record(doc,i))
 pool.sort(key=lambda x:hashlib.sha256(('ouro-weight-repair-20261103|'+x['id']).encode()).hexdigest())
 discovery=pool[:192];selection=pool[192:384];assert len(selection)==len(discovery)==192
 assert not {fingerprint(x['question']) for x in discovery}&{fingerprint(x['question']) for x in selection}
 cohorts={name:render(rows,demos) for name,rows in [('discovery',discovery),('selection',selection)]}
 audit={name:write(name+'.jsonl',rows) for name,rows in cohorts.items()}
 save('data/fixed_demonstrations.json',dict(train_indices=indices,ids=[d['id'] for d in demos],documents=demos,
    rule='random.Random(2021).sample(canonical_train,25), fixed order, harness first_n; replaces legacy advancing sampler'))
 shared=engine(demos,[discovery[0]['source_doc']]).sampler.get_context(discovery[0]['source_doc'],25)
 assert all(x['prompt'].startswith(shared) for values in cohorts.values() for x in values)
 (ROOT/'data/shared_demo_prefix.txt').write_text(shared,encoding='utf8')
 save('data/TRAIN_MANIFEST_AUDIT.json',dict(status='passed',official_train_n=1119,demonstration_count=25,eligible_unique_n=len(pool),
    discovery_n=192,selection_n=192,initial_n_each=64,cohorts=audit,seed=2021,
    train_sha256=hashlib.sha256((ROOT/'data/raw/train.parquet').read_bytes()).hexdigest(),
    validation_or_test_opened=False,legacy_demo_union_size_1172_queries=1119,
    protocol_change='Fixed25-shot demonstrations are identical for every new method; no claim of exact old per-question prompt parity'))
 print(json.dumps(dict(status='passed',cohorts=audit,eligible_unique_n=len(pool),demonstration_count=25)),flush=True)
if __name__=='__main__':main()
