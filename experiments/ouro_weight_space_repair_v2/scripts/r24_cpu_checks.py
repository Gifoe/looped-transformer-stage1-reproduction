"""R24 synthetic-only boundary/df/set/ordering checks; no real support extraction."""
from r24_support import accepted_positions,specific_sets,candidate_support

def main():
 legal,cross=accepted_positions([(0,1),(1,3),(3,5),(5,7),(7,9)],2,7);assert legal==[2,3] and cross==[1]
 try:accepted_positions([(0,3)],0,3)
 except ValueError:pass
 else:raise AssertionError('Missing predecessor accepted')
 entries=[dict(value_id=v,demo_id=d) for v,d in [(1,0),(1,0),(1,1),(2,0),(2,1),(2,2),(3,0),(4,1),(5,2),(6,3)]];decode={1:' ab ',2:'word',3:' ! ',4:'a',5:'中文',6:'12'}
 a,s,df,excluded=specific_sets(entries,decode.__getitem__);assert s=={1,5,6} and df[1]==2 and df[2]==3
 duplicated=specific_sets(entries+[entries[0]],decode.__getitem__);assert duplicated[:3]==(a,s,df)
 targets=[[1,2,1],[3,5],[6,5]];out=candidate_support(targets,0,a,s);assert out[0]['specific_difference_vs_R6']==[] and out[1]['specific_difference_vs_R6']==[5]
 order=[2,0,1];permuted=candidate_support([targets[i] for i in order],1,a,s)
 for j,old in enumerate(order):assert {k:v for k,v in permuted[j].items() if k!='candidate'}=={k:v for k,v in out[old].items() if k!='candidate'}
 return dict(status='passed_CPU_synthetic',offset_full_containment=True,boundary_crossing_rejected=True,token0_without_predecessor_rejected=True,duplicate_occurrence_set_invariant=True,demo_document_frequency=True,unicode_alnum_whitespace_punctuation_rules=True,candidate_permutation_target_mapping=True,R6_anchor_zero_difference=True,scope='Synthetic only, no real576 support/tokenizer/model/state/GPU')
if __name__=='__main__':
 import json
 print(json.dumps(main()))
