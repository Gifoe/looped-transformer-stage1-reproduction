"""Fail-closed CPU partition of actual saved offsets; no model execution."""
import json,hashlib
if not __debug__:raise RuntimeError("optimized Python forbidden")

def partition(metadata_bytes, expected_metadata_sha256, ledger):
 if hashlib.sha256(metadata_bytes).hexdigest()!=expected_metadata_sha256:raise ValueError('metadata hash')
 m=json.loads(metadata_bytes);assert m['id']==ledger['id'] and m['split']==ledger['split']
 records=sorted(m['records'],key=lambda r:r['candidate']);assert records and [r['candidate'] for r in records]==list(range(len(records)))
 groups=ledger['groups'];assert set(groups)=={'prefix','current','crossing','zero_offset'} and not groups['crossing'] and not groups['zero_offset']
 first=records[0]['encoding'];L=ledger['input_length'];bound=ledger['prefix_chars'];ids=first['input_ids']; offsets=first['offset_mapping'];readouts=first['predecessor_positions'];assert len(ids)==len(offsets)==L and len(readouts)==1 and readouts==[ledger['readout_position']]
 assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==ledger['input_sha256']
 for r in records:
  e=r['encoding'];assert e['input_ids']==ids and e['offset_mapping']==offsets and e['predecessor_positions']==readouts and len(e['target_ids'])==1 and r['prompt_sha256']==ledger['prompt_sha256']
 reconstructed={'prefix':[],'current':[],'crossing':[],'zero_offset':[]}
 for i,(a,b) in enumerate(offsets):
  if b<=a:reconstructed['zero_offset'].append(i)
  elif b<=bound:reconstructed['prefix'].append(i)
  elif a>=bound:reconstructed['current'].append(i)
  else:reconstructed['crossing'].append(i)
 assert reconstructed==groups and not reconstructed['crossing'] and not reconstructed['zero_offset'];P=set(groups['prefix']);R=set(readouts);Q=set(groups['current'])-R
 assert R<=set(groups['current']) and not(P&Q or P&R or Q&R) and P|Q|R==set(range(L))
 return {'P':sorted(P),'Q':sorted(Q),'R':sorted(R)}
