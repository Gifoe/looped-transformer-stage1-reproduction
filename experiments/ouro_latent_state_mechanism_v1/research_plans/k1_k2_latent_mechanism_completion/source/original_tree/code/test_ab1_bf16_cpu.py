import hashlib,torch
from ab1_boundary_core import corner,q9_cross,control
E=torch.tensor([[[0.,0.],[.25,-.5],[1.,-2.],[0.,0.],[3.,4.]]],dtype=torch.bfloat16);B=torch.tensor([[[.5,1.],[.125,.25],[-2.,1.],[1.,1.],[6.,8.]]],dtype=torch.bfloat16);P=2;R=4
for qs in ("E","B"):
 for rs in ("E","B"):
  x=corner(E,B,P,R,qs,rs);assert x.dtype==torch.bfloat16 and torch.equal(x[:,:P],B[:,:P]) and torch.equal(x[:,P:R],(E if qs=="E" else B)[:,P:R]) and torch.equal(x[:,R],(E if rs=="E" else B)[:,R])
base=corner(E,B,P,R,"E","B");donor=corner(E,B,P,R,"B","B");a,r=control(base,donor,P,R,"Q","random","train","fixed-id","Q@R=B");b,s=control(base,donor,P,R,"Q","random","train","fixed-id","Q@R=B");assert torch.equal(a,b) and all(torch.equal(x["cast_residual"],y["cast_residual"]) for x,y in zip(r["rows"],s["rows"]));assert torch.equal(a[:,:P],base[:,:P]) and torch.equal(a[:,R:],base[:,R:]);assert all(abs(x['actual_displacement']-x['nominal_displacement'])<.03 for x in r['rows'])
print('PASS BF16 four corners, deterministic per-token CPU generator, non-R boundary and measured quantization; torch',torch.__version__,'output_sha256',hashlib.sha256(a.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
