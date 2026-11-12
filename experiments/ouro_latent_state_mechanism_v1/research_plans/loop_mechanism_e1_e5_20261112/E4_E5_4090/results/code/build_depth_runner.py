"""Generate a separate late-depth collector from the immutable Order004 source."""
from pathlib import Path
import hashlib,json
p=Path(__file__).resolve().parent;source=(p/'state_mediation_runner.py').read_text();s=source
changes=[
 ('Order004: actual L18 R-state block and attention mediator factorial.','Order005: matched late-depth L18 R-state and attention mediator factorial.'),
 ('ROOT=Path(__file__).resolve().parent','ROOT=Path(__file__).resolve().parent\nRECEIVER=2  # Required CLI --receiver selects 3 or 4 before any model operation.'),
 ("input_r_block(model,2,q['R'],x)","input_r_block(model,RECEIVER,q['R'],x)"),
 ("r.run(model,q,2,patch=a,capture=capture)","r.run(model,q,RECEIVER,patch=a,capture=capture)"),
 ("p['native']['2']","p['native'][str(RECEIVER)]"),
 ("pt['K2_'+name]","pt[f'K{RECEIVER}_'+name]"),
 ("pt['K1_x_R']","pt[f'K{RECEIVER-1}_x_R']"),
 ("pt['K1_O']","pt[f'K{RECEIVER-1}_O']"),
 ("'native':{'2':n}","'native':{str(RECEIVER):n}"),
 ("'reused_K1_native':p['native']['1']","'reused_K1_native':p['native']['1'],'reused_donor_native':p['native'][str(RECEIVER-1)],'receiver_loop':RECEIVER,'donor_loop':RECEIVER-1,'X1_A1_alias_definition':'X1/A1 mean preceding native donor; reused_K1_native remains true global K1'"),
 ("m['native']['2']","m['native'][str(RECEIVER)]"),
 ("'pair':'1-2'","'pair':f'{RECEIVER-1}-{RECEIVER}'"),
 ("'receiver_loop':2","'receiver_loop':RECEIVER"),
 ("'donor_loop':1","'donor_loop':RECEIVER-1"),
 ('def main():','def main():\n global RECEIVER'),
 ("a.add_argument('--limit',type=int);args=a.parse_args();out=Path(args.output)","a.add_argument('--limit',type=int);a.add_argument('--receiver',required=True,type=int,choices=[3,4]);args=a.parse_args();RECEIVER=args.receiver;out=Path(args.output)"),
 ("precision=sorted({i for v in groups.values() for i in v[:2]}.intersection(ordered))","precision=sorted(set(sorted(cohort['ids'])[:8]).intersection(ordered))"),
 ("'experiment':'ASTRA_ORDER004_R_STATE_ATTENTION_MEDIATION'","'experiment':'ASTRA_ORDER005_MATCHED_LATE_R_STATE_ATTENTION_MEDIATION','receiver_loop':RECEIVER,'donor_loop':RECEIVER-1,'precision_rule':'FIRST_EIGHT_LEXICOGRAPHIC_VALIDATION_IDS','X1_A1_alias_definition':'preceding native donor; global K1 is separately reused_K1_native'"),
]
counts=[]
for before,after in changes:
 count=s.count(before);assert count>0,before;counts.append({'before':before,'after':after,'count':count});s=s.replace(before,after)
target=p/'depth_state_mediation_runner.py';target.write_bytes(s.replace('\n','\r\n').encode())
(p/'depth_runner_generation_receipt.json').write_text(json.dumps({'hash_scope':'RAW_FILE_BYTES','source_sha256':hashlib.sha256((p/'state_mediation_runner.py').read_bytes()).hexdigest(),'generated_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'generated_normalized_LF_sha256':hashlib.sha256(s.encode()).hexdigest(),'line_endings':'Explicit CRLF, matching the executed frozen producer on both Windows and Linux','changes':counts},indent=2))
print(target)
