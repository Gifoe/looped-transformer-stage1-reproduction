"""Strict early-rescue persistence; derived from the same frozen question results."""
import pathlib,json,os
R=pathlib.Path(os.environ.get('CCRU_ROOT','/root/ouro-context-update-sol-20261117'));rows=[json.loads(l) for l in (R/'ALL_QUESTION_RESULTS.jsonl').read_text().splitlines()];selected=json.loads((R/'SELECTED_METHOD.json').read_text())['method'];out={}
for cohort in ['development','confirmation']:
 records=[]
 for q in rows:
  if q['cohort']!=cohort:continue
  a=q['conditions'][selected]['scores'];b=q['conditions']['N']['scores'];g=q['gold'];strict=lambda d:len(a[d]['ties'])==len(b[d]['ties'])==1;early=a['2']['prediction']==g and b['2']['prediction']!=g and strict('2')
  if early:records.append({'id':q['id'],'native_K4_correct':b['4']['prediction']==g,'modified_K3_correct':a['3']['prediction']==g,'modified_K4_correct':a['4']['prediction']==g,'becomes_new_strict_K4_rescue':a['4']['prediction']==g and b['4']['prediction']!=g and strict('4')})
 out[cohort]={'strict_K2_rescues':len(records),'still_correct_at_K4':sum(z['modified_K4_correct'] for z in records),'new_strict_K4_rescues':sum(z['becomes_new_strict_K4_rescue'] for z in records),'rows':records}
(R/'STRICT_PROPAGATION.json').write_text(json.dumps(out,indent=2));print('STRICT_PROPAGATION',out)
