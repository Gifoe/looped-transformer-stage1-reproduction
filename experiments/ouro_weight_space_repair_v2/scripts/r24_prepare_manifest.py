"""Prepare metadata-only sanitized requests and known25demo answers; no supports."""
import json,numpy as np
from r24_common import ROOT,read,save,sha

def main():
 rows=[];states={r['id']:r for r in read('results/slcd_SOURCE_AUDIT.json')['states']}
 for split,n in [('discovery',384),('selection',192)]:
  data=[json.loads(x) for x in (ROOT/('data/slcd_'+split+'.jsonl')).read_text(encoding='utf8').splitlines()];old={r['id']:r for r in [json.loads(x) for x in (ROOT/('results/r7_'+split+'_development.jsonl')).read_text(encoding='utf8').splitlines()]};assert len(data)==len(old)==n
  done=read('results/r7_'+split+'_development_COMPLETE.json');assert done['status']=='complete' and done['n']==n and done['score_file_sha256']==sha('results/r7_'+split+'_development.jsonl') and done['provenance']['data_sha256']==sha('data/slcd_'+split+'.jsonl')
  assert all(r['signature']==done['signature'] for r in old.values())
  for item in data:
   record=old[item['id']];assert item['prompt_sha256']==record['prompt_sha256'] and item['choice_char_lengths']==record['choice_char_lengths'];state=states[item['id']]
   rows.append(dict(id=item['id'],split=split,prompt=item['prompt'],current_question=item['source_doc']['question'],request_pairs=item['request_pairs'],choice_texts=item['choices'],choice_char_lengths=item['choice_char_lengths'],choice_token_counts=state['choice_token_counts'],target_id_sha256=state['target_id_sha256'],R6_winner=int(np.argmax(record['baseline_scores']['M0_R6']))))
 docs=read('data/slcd_fixed_demonstrations.json')['documents'];assert len(docs)==25
 demos=[dict(id=d['id'],question=d['question'],answer=d['choices']['text'][d['choices']['label'].index(d['answerKey'])]) for d in docs]
 save('configs/r24_INPUT_MANIFEST.json',dict(n=576,rows=rows,demonstrations=demos,scope='Sanitized no current gold; R6 winner and original requests metadata only. No tokenizer or support computed.'))
 print('R24_METADATA_MANIFEST_PREPARED',576)
if __name__=='__main__':main()
