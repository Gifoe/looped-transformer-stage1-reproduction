import hashlib,json,copy
from partition import partition
ids=[11,12,13,14];enc={'input_ids':ids,'offset_mapping':[[0,2],[2,4],[4,5],[5,7]],'predecessor_positions':[3],'target_ids':[8]};r={'candidate':0,'encoding':enc,'prompt_sha256':'x'};m={'id':'a','split':'train','records':[r]};ledger={'id':'a','split':'train','prefix_chars':4,'input_length':4,'readout_position':3,'prompt_sha256':'x','input_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),'groups':{'prefix':[0,1],'current':[2,3],'crossing':[],'zero_offset':[]}}
def run(m,l):
 b=json.dumps(m).encode();return partition(b,hashlib.sha256(b).hexdigest(),l)
assert run(m,ledger)=={'P':[0,1],'Q':[2],'R':[3]}
for mutation in ('cross','zero','id','readout','duplicate'):
 x=copy.deepcopy(m);l=copy.deepcopy(ledger)
 if mutation=='cross':x['records'][0]['encoding']['offset_mapping'][1]=[2,5]
 if mutation=='zero':x['records'][0]['encoding']['offset_mapping'][1]=[2,2]
 if mutation=='id':l['id']='wrong'
 if mutation=='readout':x['records'][0]['encoding']['predecessor_positions']=[1];l['readout_position']=1
 if mutation=='duplicate':x['records'].append(copy.deepcopy(x['records'][0]))
 try:run(x,l)
 except (AssertionError,ValueError):pass
 else:raise AssertionError('accepted '+mutation)
print('PARTITION_SYNTHETIC_CPU_PASS')
