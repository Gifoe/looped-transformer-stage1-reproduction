"""Fork frozen operators only for explicit preregistered precision IDs and scope names."""
from pathlib import Path
import json,hashlib
p=Path(__file__).resolve().parent;report=[]
for source,target,experiment in [('query_factorial_runner.py','confirmation_query_factorial_runner.py','ASTRA_ORDER006_PREREGISTERED_NATURAL_K2_QUERY_FACTORIAL'),('depth_state_mediation_runner.py','confirmation_state_mediation_runner.py','ASTRA_ORDER006_PREREGISTERED_R_STATE_ATTENTION_MEDIATION')]:
 text=(p/source).read_text();old=text
 group=" groups={}\n for q in cohort.get('questions',[]):groups.setdefault(q['historical_group'],[]).append(q['id'])\n"
 assert text.count(group)==1; text=text.replace(group,'')
 if source.startswith('query'):
  text=text.replace('Astra Order003: natural K2 Q x KV finite differences, reused audited KV cell.','Order006 preregistered natural K2 Q x KV confirmation, with reduced-base KV reuse.')
  before="precision=sorted({ident for v in groups.values() for ident in v[:2]}.intersection(ordered))";after="precision=sorted(set(cohort['precision_ids']).intersection(ordered))";assert before in text;text=text.replace(before,after)
  text=text.replace('ASTRA_ORDER003_NATURAL_K2_QUERY_FACTORIAL',experiment)
 else:
  text=text.replace('Order005: matched late-depth L18 R-state and attention mediator factorial.','Order006 preregistered L18 R-state and actual-attention mediation confirmation.').replace('Required CLI --receiver selects 3 or 4','Required CLI --receiver selects 2, 3 or 4')
  before="precision=sorted(set(sorted(cohort['ids'])[:8]).intersection(ordered))";after="precision=sorted(set(cohort['precision_ids']).intersection(ordered))";assert before in text;text=text.replace(before,after)
  text=text.replace('choices=[3,4]','choices=[2,3,4]').replace('ASTRA_ORDER005_MATCHED_LATE_R_STATE_ATTENTION_MEDIATION',experiment).replace('FIRST_EIGHT_LEXICOGRAPHIC_VALIDATION_IDS','FIRST_EIGHT_LEXICOGRAPHIC_PREREGISTERED_CONFIRMATION_IDS')
 (p/target).write_bytes(text.encode());report.append({'source':source,'source_raw_sha256':hashlib.sha256((p/source).read_bytes()).hexdigest(),'output':target,'output_raw_sha256':hashlib.sha256((p/target).read_bytes()).hexdigest(),'scope':'No operator arithmetic or hook changed; explicit frozen precision IDs, experiment name, and receiver2 support for generic R collector only.'})
(p/'confirmation_runner_generation_receipt.json').write_bytes(json.dumps(report,indent=2).encode());print(json.dumps(report,indent=2))
