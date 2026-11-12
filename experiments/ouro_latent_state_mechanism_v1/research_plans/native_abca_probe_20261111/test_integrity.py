import copy,torch
from observer import observe
from integrity import validate_observations,validate_records,compare,digest_ids
from test_observer import Model
x=torch.randn(1,4,4);m=Model()
with observe(m,3) as r:m(x)
expected={'head_dim':4,'hidden_size':4,'q_heads':1,'kv_heads':1,'dtype':'torch.float32','device':'cpu'}
assert validate_observations(r,3,4,expected)
for f in ('is_causal','attn_mask','completed'):
 v=copy.deepcopy(r);v['sdpa'][0][f]={'is_causal':False,'attn_mask':'UNKNOWN','completed':False}[f]
 try:validate_observations(v,3,4,expected)
 except ValueError:pass
 else:raise AssertionError(f)
v=copy.deepcopy(r);v['rope'][0]['position_ids'][0,0]=99
try:validate_observations(v,3,4,expected)
except ValueError:pass
else:raise AssertionError('position')
for mutation in ('empty_rope','one_rope','wrong_klen','wrong_hidden','wrong_dtype','wrong_device'):
 v=copy.deepcopy(r)
 if mutation=='empty_rope':v['rope'][0]['cos_sin']=()
 elif mutation=='one_rope':v['rope'][0]['cos_sin']=v['rope'][0]['cos_sin'][:1]
 elif mutation=='wrong_klen':v['sdpa'][0]['qkv'][1]['shape'][2]=3
 elif mutation=='wrong_hidden':v['states'][0]['prefix']=v['states'][0]['prefix'][:,:,:3]
 elif mutation=='wrong_dtype':v['sdpa'][0]['qkv'][0]['dtype']='torch.float16'
 else:v['sdpa'][0]['qkv'][0]['device']='cuda:0'
 try:validate_observations(v,3,4,expected)
 except ValueError:pass
 else:raise AssertionError(mutation)
ids=[1,2,3,4];record={'candidate':0,'encoding':{'input_ids':ids,'predecessor_positions':[3],'target_ids':[5]},'scores':[0.,1.]}
sel={'input_length':4,'input_sha256':digest_ids(ids)}
assert validate_records([record],sel,[1,2])[0]==ids
try:validate_records([record],sel,[1,3])
except ValueError:pass
else:raise AssertionError('prefix')
a=torch.tensor([0.,1.],dtype=torch.bfloat16);b=torch.tensor([-0.,2.],dtype=torch.bfloat16);c=compare(a,b)
assert not c['bitwise_equal'] and c['unequal_elements']==1 and c['max_absolute']==1
print('INTEGRITY_TOY_PASS; NOT GPU')
