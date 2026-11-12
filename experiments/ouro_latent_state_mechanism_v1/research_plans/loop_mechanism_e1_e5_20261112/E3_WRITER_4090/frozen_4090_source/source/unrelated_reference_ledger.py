"""Deterministic same-site unrelated directions from pre-treatment N/P reference pool."""
import hashlib,torch
if not __debug__:raise RuntimeError('optimized Python forbidden')
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def unrelated_directions(recipient,pool):
 """pool records have id/input_ids_sha256/deltas[(physical_index,branch)] only; no labels."""
 assert len(pool)==64 and len({r['id'] for r in pool})==64
 sites={(i,b) for i in range(8,16) for b in ('a','b')};assert set(recipient['deltas'])==sites and all(set(x['deltas'])==sites for x in pool)
 candidates=sorted((x for x in pool if x['id']!=recipient['id'] and x['input_ids_sha256']!=recipient['input_ids_sha256']),key=lambda x:hashlib.sha256(('q-to-r-interval-v1|'+recipient['id']+'|'+x['id']).encode()).digest());mapping={};vectors={}
 for site in sorted(sites):
  ref=recipient['deltas'][site];assert ref.dtype==torch.float64 and ref.shape==(1,2048) and torch.isfinite(ref).all();dose=ref.norm();tag=f'{site[0]:02d}_{site[1]}'
  if dose==0:
   u=torch.zeros_like(ref);mapping[tag]={'status':'ZERO_REFERENCE_IDENTITY','donor_id':None,'reference_sha256':digest(ref),'reference_dose':0.0,'unrelated_sha256':digest(u)};vectors[site]=u;continue
  chosen=None
  for donor in candidates:
   d=donor['deltas'][site];assert d.dtype==torch.float64 and d.shape==ref.shape and torch.isfinite(d).all()
   if d.norm()>0:chosen=donor;break
  if chosen is None:mapping[tag]={'status':'UNDEFINED_NO_NONZERO_DONOR','donor_id':None,'reference_sha256':digest(ref),'reference_dose':float(dose)};vectors[site]=None;continue
  d=chosen['deltas'][site];u=d*(dose/d.norm());assert torch.isfinite(u).all() and abs(float(u.norm()-dose))<=1e-10*max(1.0,float(dose));vectors[site]=u;mapping[tag]={'status':'OK','donor_id':chosen['id'],'donor_input_ids_sha256':chosen['input_ids_sha256'],'reference_sha256':digest(ref),'donor_delta_sha256':digest(d),'unrelated_sha256':digest(u),'reference_dose':float(dose),'donor_original_dose':float(d.norm()),'unrelated_dose':float(u.norm())}
 return mapping,vectors
