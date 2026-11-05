"""Native-only immutable task stream and unchanged original-position reduction."""
from r28_common import read
from r28_stream import stream_contract,candidate_readouts as old_candidate_readouts
METHODS=['primary','J2_guidance','M2','M3','G_matched','vocabulary_null','LOO_N0','LOO_N1','LOO_N2','Fixed','matched_scale','reference_consensus_matched','native1','native4','native5','adaptive','N0_native1','N0_native4','N0_Fixed']
STRONG_CONTROLS=['M3','J2_guidance','G_matched','vocabulary_null','matched_scale','reference_consensus_matched']
def native_stream():
 graph,tasks,uses,checks=stream_contract();native={k:t for k,t in tasks.items() if t['mode']=='native'};groups=[dict(reference_tasks={n+'_native':g['reference_tasks'][n+'_native'] for n in ['N0','N1','N2']},Q_consumers=list(g['Q_consumers'])) for g in graph['stream_groups']];observed=[k for g in groups for k in list(g['reference_tasks'].values())+g['Q_consumers']];assert len(observed)==len(set(observed))==len(native)==1588 and set(observed)==set(native)
 return graph,native,uses,groups,checks['exact_duplicate_Q_task']
def candidate_readouts(rows,finished):
 # The old helper preserves every original Torch-sum/character/late/R6 operator.
 result=old_candidate_readouts(rows,finished)
 for rec,row in zip(result,rows):
  for c,ch in zip(rec['candidates'],row['candidates']):
   for name in ['LOO_N0','reference_consensus_matched']:
    p=finished[(row['id'],ch['candidate'],name)];c['pieces'][name]=p;c['scores'][name]=p['sum_score']/ch['characters']+c['late']
   assert set(c['pieces'])==set(METHODS)
 return result
