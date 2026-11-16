import core,json,random,hashlib,datetime
from transformers import AutoTokenizer
R=core.ROOT;tok=AutoTokenizer.from_pretrained(core.MODEL,local_files_only=True,trust_remote_code=True);rng=random.Random(20261116)
labels=list('ABCD');targets=[tok.encode(' '+c,add_special_tokens=False) for c in labels];assert all(len(t)==1 for t in targets);targets=[t[0] for t in targets];assert targets==[330,389,340,422],targets
prefix='Follow the stated containment relations. Select the final cabinet.\n\nObject X is in Box M.\nBox M is in Cabinet 5.\nBox N is in Cabinet 6.\nQuestion: Which cabinet contains object X?\nA. Cabinet 6\nB. Cabinet 5\nC. Cabinet 7\nD. Cabinet 8\nAnswer: B\n\nObject Y is in Box N.\nBox M is in Cabinet 7.\nBox N is in Cabinet 8.\nQuestion: Which cabinet contains object Y?\nA. Cabinet 5\nB. Cabinet 6\nC. Cabinet 8\nD. Cabinet 7\nAnswer: C\n\n'
pairs=[];used=set();rows=[]
def build(pid,split,template,obj,boxes,values,order,rev):
 a,b,c,d=values;ba,bb=boxes;out={};opt=list(reversed(order)) if rev else order
 for world,variable,maps in [('original',ba,(a,b)),('donor',bb,(c,d)),('original_edit',bb,(a,b)),('donor_edit',ba,(c,d))]:
  if template==0:
   clause=f'Object {obj} is in Box {variable}.\n';mapping=f'Box {ba} is in Cabinet {maps[0]}.\nBox {bb} is in Cabinet {maps[1]}.\n';question=f'Question: Which cabinet contains object {obj}?\n'
  else:
   clause=f'The box containing object {obj} is Box {variable}.\n';mapping=f'Box {ba} is located inside Cabinet {maps[0]}.\nBox {bb} is located inside Cabinet {maps[1]}.\n';question=f'Question: Identify the cabinet containing object {obj}.\n'
  text=prefix+clause+mapping+question+''.join(f'{lab}. Cabinet {val}\n' for lab,val in zip(labels,opt))+'Answer:'
  enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);ids=enc['input_ids'];offsets=enc['offset_mapping'];start=len(prefix)+clause.index('Box '+variable);end=start+len('Box '+variable)
  span=[i for i,(s,e) in enumerate(offsets) if e>start and s<end];assert span==list(range(span[0],span[-1]+1));mapping_start=min(i for i,(s,e) in enumerate(offsets) if e>len(prefix)+len(clause))
  upos=text.index('Cabinet '+str(maps[0]),len(prefix)+len(clause))+len('Cabinet ');unrel=[i for i,(s,e) in enumerate(offsets) if e>upos and s<upos+1];assert len(unrel)==1
  result=maps[0] if variable==ba else maps[1];q={'id':f'{pid}_L{int(rev)}_{world}','pair_id':pid,'split':split,'template':template,'reverse_labels':rev,'world':world,'text':text,'ids':ids,'R':len(ids)-1,'targets':targets,'prefix_len':mapping_start-len(tok(clause,add_special_tokens=False)['input_ids']),'source_span':[span[0],span[-1]+1],'mapping_start':mapping_start,'unrelated_span':[unrel[0],unrel[0]+1],'input_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'correct_index':opt.index(result),'correct_cabinet':result,'options':opt,'intermediate':variable,'mappings':dict(zip(boxes,maps))}
  assert q['source_span'][1]<=mapping_start<q['R'];out[world]=q;rows.append(q)
 for receiver,donor,edit,same,cross in [('original','donor','original_edit','donor_edit',b),('donor','original','donor_edit','original_edit',c)]:
  qr=out[receiver];qd=out[donor];qe=out[edit];qs=out[same]
  assert len(qr['ids'])==len(qd['ids'])==len(qe['ids'])==len(qs['ids']) and qr['source_span']==qd['source_span']==qe['source_span']==qs['source_span'] and qr['unrelated_span']==qd['unrelated_span']
  assert len({qr['correct_index'],qd['correct_index'],qr['options'].index(cross),next(i for i in range(4) if i not in [qr['correct_index'],qd['correct_index'],qr['options'].index(cross)])})==4
  for q in (qr,qd,qe,qs):assert q['ids'][:qr['source_span'][0]]==qr['ids'][:qr['source_span'][0]]
  pairs.append({'unit_id':f'{pid}_L{int(rev)}_{receiver}','pair_id':pid,'split':split,'template':template,'reverse_labels':rev,'receiver':qr['id'],'donor':qd['id'],'text_edit':qe['id'],'same_variable':qs['id'],'predicted_cross_index':qr['options'].index(cross),'original_index':qr['correct_index'],'donor_index':qd['correct_index'],'predicted_cross_cabinet':cross})
for i in range(160):
 split='development' if i<32 else 'heldout';template=0 if i<96 else 1
 while True:
  obj=rng.choice(list('PQRS') if split=='development' else list('TUVW'));boxes=rng.sample(list('ABCDEFGH'),2);values=rng.sample([1,2,3,4],4);order=rng.sample([1,2,3,4],4);key=(obj,tuple(boxes),tuple(values))
  if key not in used:used.add(key);break
 for rev in (False,True):build(f'CW{i:03d}',split,template,obj,boxes,values,order,rev)
(R/'B_INPUTS.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows));(R/'B_UNITS.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in pairs))
freeze={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'seed':20261116,'base_pairs':160,'development_pairs':32,'heldout_pairs':128,'heldout_template0_pairs':64,'heldout_template1_pairs':64,'units_per_basepair':4,'worlds_per_basepair':8,'new_entities':'development PQRS vs heldout TUVW; boxes and all role-to-cabinet mappings permuted; combinations unique','source':'complete Box+name token span in initial first-hop clause, precedes all mappings; actual causal mask audited','native_and_text':'four actual worlds per label permutation; correct answers verified by symbolic containment; preserve native Joint Option label scoring','conditions':['source_boundary','same_variable','unrelated','dose','readout_copy','descendant_edit'],'descendant_alternative':'exactly one preregistered alternative: receiver textual-firsthop-edit K1 B[source_start:R], original R. Broad causal descendant diagnostic; later rows can encode answer. Not localized-variable proof','statistics':'5000 base-pair-cluster bootstrap seed20261116, directions and reversed labels clustered; primary heldout source cross margin relative to each same-variable/unrelated/dose/readout control; Bonferroni4 comparisons98.75%; secondary cross accuracies descriptive95%','positive_gate':'both heldout templates and both directions/label orientations: source crossaccuracy>=0.5 and positive cross-minus-bestother mean; pooled paired margin gain over each4controls has98.75%CI lower>0; native/text baselines all reported; no native-correct filtering','primary_population':'all heldout512 directed label units; development128 separate; no posttreatment filtering','sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [R/'B_INPUTS.jsonl',R/'B_UNITS.jsonl']}}
core.jsave(R/'B_FREEZE.json',freeze);print(json.dumps({'rows':len(rows),'units':len(pairs),'targets':targets,'T_min':min(len(q['ids']) for q in rows),'T_max':max(len(q['ids']) for q in rows),'first_span':rows[0]['source_span']}))
