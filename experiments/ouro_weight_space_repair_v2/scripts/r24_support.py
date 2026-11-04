"""R24 pure set/offset rules. No labels, model, state, or learned weights."""
import unicodedata

def accepted_positions(offsets,start,end):
 if not (0<=start<end):raise ValueError('Invalid answer character interval')
 legal=[];boundary=[]
 for i,(a,b) in enumerate(offsets):
  if a<b and a<end and b>start:
   if start<=a and b<=end:
    if i==0:raise ValueError('Value token has no predecessor key position')
    legal.append(i)
   else:boundary.append(i)
 return legal,boundary

def specific_sets(entries,decode):
 all_values={e['value_id'] for e in entries};demofrequency={v:len({e['demo_id'] for e in entries if e['value_id']==v}) for v in all_values};specific=set();excluded=[]
 for v in sorted(all_values):
  text=decode(v).strip();alnum=sum(c.isalnum() for c in text);reasons=[]
  if demofrequency[v]>2:reasons.append('demo_df_gt2')
  if alnum<2:reasons.append('fewer_than2_unicode_alnum')
  if reasons:excluded.append(dict(value_id=v,decoded_trim=text,demo_df=demofrequency[v],unicode_alnum=alnum,reasons=reasons))
  else:specific.add(v)
 return all_values,specific,demofrequency,excluded

def candidate_support(targets,anchor,all_values,specific):
 if not 0<=anchor<len(targets):raise ValueError('Invalid R6 candidate index')
 sets=[set(x) for x in targets];base=sets[anchor]
 return [dict(candidate=j,target_set=sorted(t),all_intersection=sorted(t&all_values),specific_intersection=sorted(t&specific),all_difference_vs_R6=sorted((t-base)&all_values),specific_difference_vs_R6=sorted((t-base)&specific)) for j,t in enumerate(sets)]
