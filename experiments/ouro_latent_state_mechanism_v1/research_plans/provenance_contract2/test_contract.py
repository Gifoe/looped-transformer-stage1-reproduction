import tempfile,pathlib,json,torch
from contract import Toy,signature,execute,write_pair,read_pair,identity_compatible,canonical,digest
torch.set_num_threads(1);torch.manual_seed(71)
m=Toy();ids=torch.tensor([[1,2,3,4]]);mask=torch.ones_like(ids);pos=torch.arange(4).reshape(1,4)
manifest={'id':'toy-Q1','ids':signature(ids),'mask':signature(mask),'positions':signature(pos),'targets':[4,7],'readouts':[3]}
r,t,y=execute(m,ids,mask,pos,manifest,2,(1,1));assert r['receiver']=='natural' and r['status']=='SEALED'
r2,t2,y2=execute(m,ids,mask,pos,manifest,2,(1,1));assert identity_compatible(r,r2) and torch.equal(t,t2)
action={'position':0,'delta':1.};h,ht,hy=execute(m,ids,mask,pos,manifest,2,(1,1),action);h2,ht2,_=execute(m,ids,mask,pos,manifest,2,(1,1),action)
assert h['receiver']=='hybrid' and not identity_compatible(r,h) and identity_compatible(h,h2) and torch.equal(ht,ht2)
pre,pret,_=execute(m,ids,mask,pos,manifest,2,(0,1),action);plain,plaint,_=execute(m,ids,mask,pos,manifest,2,(0,1))
assert pre['receiver']=='natural' and pre['capture_lineage']==[] and len(pre['forward_lineage'])==1 and torch.equal(pret,plaint) and identity_compatible(pre,plain)
assert not torch.equal(ht,t)
newpos=pos.clone();newpos[0,1]+=1;bad=dict(manifest);bad['positions']=signature(newpos);p,_,_=execute(m,ids,mask,newpos,bad,2,(1,1));assert p['effective']['cos']!=r['effective']['cos'] and not identity_compatible(r,p)
newmask=mask.clone();newmask[0,3]=0;bad=dict(manifest);bad['mask']=signature(newmask);p,_,_=execute(m,ids,newmask,pos,bad,2,(1,1));assert p['effective']['mask']!=r['effective']['mask'] and not identity_compatible(r,p)
try:execute(m,ids,mask,pos,manifest,2,(1,1),fail=True)
except RuntimeError:pass
else:raise AssertionError('failed forward was accepted')
with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).parent) as tmp:
 folder=pathlib.Path(tmp)/'case';expected=write_pair(folder,r,t);rr,tt=read_pair(folder,expected);assert rr==r and torch.equal(tt,t)
 receipt=folder/'receipt.json';original=receipt.read_bytes();receipt.write_bytes(original+b' ')
 try:read_pair(folder,expected)
 except ValueError:pass
 else:raise AssertionError('tampered receipt')
 receipt.write_bytes(original);binary=folder/'tensor.safetensors';raw=bytearray(binary.read_bytes());raw[-1]^=1;binary.write_bytes(raw)
 try:read_pair(folder,expected)
 except (ValueError,RuntimeError):pass
 else:raise AssertionError('tampered tensor')
print('SYNTHETIC_EFFECTIVE_CONTEXT_PERSISTENCE_PASS; NO AUTHENTICATED ORIGIN')
