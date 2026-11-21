"""Equal-horizon attention interventions with fixed causal support."""
import torch

from behavior import answer_tokens
from native import Recorder, pad

HEADS=(1,7,11,14)
OTHER=(0,2,3,4)
LAYER=5
SEED=21112117


def capture(f,ids,k=2,edit=None,start=None):
    r=Recorder(keys={(i,LAYER) for i in range(k)})
    h=f.run(ids,k,edit=edit,record=r,start=start)
    return h,r


class Patch:
    def __init__(self,lengths,*,a=None,v=None,heads=HEADS,pass_index=0,write=0.,writer_control=None):
        self.lengths=lengths
        self.a,self.v,self.heads=a,v,list(heads)
        self.index,self.dose,self.control=pass_index,write,writer_control

    def resid(self,k,l,x):
        return x

    def mlp(self,key,z,r):
        return z

    def attn(self,key,a,v):
        if key!=(self.index,LAYER):
            return a,v
        if self.a is not None:
            a=a.clone()
        if self.v is not None:
            v=v.clone()
        for lane,n in enumerate(self.lengths):
            if self.a is not None:
                src=self.a[lane] if self.a.shape[0]>1 else self.a[0]
                a[lane,self.heads,:n,:n]=src[:,:n,:n]
            if self.v is not None:
                src=self.v[lane] if self.v.shape[0]>1 else self.v[0]
                v[lane,self.heads,:n]=src[:,:n]
        return a,v

    def write(self,key,name,x):
        if key!=(0,0) or name!="a_write" or not self.dose:
            return x
        z=x.clone()
        for lane,n in enumerate(self.lengths):
            v=x[lane,:n]
            if self.control=="permute":
                # Same-dimension same-norm perturbation with frozen coordinates.
                v=v.roll(37,-1)
            z[lane,:n]-=self.dose*v
        return z


def lane_plan(tok,prompt,choices):
    encoded=[answer_tokens(tok,prompt,v) for v in choices]
    lens=[n for _,n in encoded]
    ids,_=pad([x for x,_ in encoded])
    coords=[(i,j-1) for i,(x,n) in enumerate(encoded) for j in range(n,len(x))]
    index=torch.tensor(coords,device=ids.device)
    targets=ids[index[:,0],index[:,1]+1]
    sizes=[len(x)-n for x,n in encoded]
    def read(model,h):
        z=model.lm_head(h[index[:,0],index[:,1]]).float()
        lp=z.log_softmax(-1).gather(-1,targets[:,None]).flatten()
        return [x.tolist() for x in lp.split(sizes)]
    return ids,lens,read


def scramble(a,b,key=SEED):
    """Randomly exchange the two original rows. Preserves causal support and exact pair dose."""
    g=torch.Generator(device=a.device).manual_seed(key)
    pick=torch.rand(a.shape[:-1],generator=g,device=a.device)[...,None]>.5
    return torch.where(pick,a,b),torch.where(pick,b,a)


def sink_mix(native,donor,kind):
    # Both patterns have the same causal support. Base first column indexes sequence position 0.
    if kind=="position0":
        mass=donor[...,:1]
        other=native[...,1:]
    elif kind=="rest":
        mass=native[...,:1]
        other=donor[...,1:]
    else:
        raise ValueError(kind)
    free=1-mass
    normal=other.sum(-1,keepdim=True)
    assert not ((normal<1e-12)&(free>1e-6)).any().item()
    b=other/normal.clamp_min(1e-12)*free
    return torch.cat((mass,b),dim=-1)


def dose(x,y):
    return float((x-y).float().norm())


def prefix_pos0(a,n):
    return float(a[0,:,:n,0].mean())


def norms(h,a,lens):
    return max(float((h[i,:n].norm(dim=-1)/a[i,:n].norm(dim=-1).clamp_min(1e-10)).max())
               for i,n in enumerate(lens))

