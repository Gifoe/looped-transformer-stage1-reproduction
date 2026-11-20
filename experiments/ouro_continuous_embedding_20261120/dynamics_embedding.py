"""Native residual operators and candidate embedding vector fields."""
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

import torch
from torch.func import jvp, vjp
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE=Path(__file__).resolve().parent
REFINE=HERE.parent/"ouro_continuum_refinement_20261120"
sys.path.insert(0,str(REFINE))
from solver import Operators, integrate_a, integrate_b, weight_hash


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(p,data):
    temp=p.with_suffix(".tmp")
    temp.write_text(json.dumps(data,allow_nan=False,ensure_ascii=False,separators=(",",":"))+"\n",encoding="utf-8")
    temp.replace(p)


def lengths(op,t):
    return [row[:n] for row,n in zip(t.detach().tolist(),op.lengths.tolist(),strict=True)]


def norm(op,h):
    return (h.float().square().sum(-1)*op.valid).sum(-1).sqrt()


def cosine(op,x,y):
    d=((x.float()*y.float()).sum(-1)*op.valid).sum(-1)
    return (d/(norm(op,x)*norm(op,y)).clamp_min(1e-12)).clamp(-1,1)


def rel(op,x,y):
    return norm(op,x-y)/norm(op,y).clamp_min(1e-10)


def state_error(op,x,y):
    return {"full":rel(op,x,y).tolist(),
        "tokens":lengths(op,(x.float()-y.float()).norm(dim=-1)/y.float().norm(dim=-1).clamp_min(1e-10))}


def compare(op,h,ref,start):
    a,b=op.scores(h),op.scores(ref)
    return dict(error=state_error(op,h,ref),update_cos=cosine(op,h-start,ref-start).tolist(),
        max_logit_error=(a-b).abs().amax(dim=-1).tolist(),pred=a.argmax(-1).tolist(),
        ref_pred=b.argmax(-1).tolist(),logits=a.tolist(),ref_logits=b.tolist())


def noise(op,h,seed):
    gen=torch.Generator(device=h.device).manual_seed(seed)
    return torch.randn(h.shape,device=h.device,generator=gen)*op.valid.unsqueeze(-1)


class EmbeddingOperators(Operators):
    def __init__(self,model,items,depth):
        super().__init__(model,items,depth=depth)
        self.counts.update(jvp=0,vjp=0,inverse_iterations=0)

    def native_suffix(self,h,start=0):
        x=h.to(self.dtype)
        for i in range(start,48):
            x=x+self.branch(i,x)
        return self.norm(x)

    def deriv(self,field,x,v):
        self.counts["jvp"]+=1
        with sdpa_kernel(SDPBackend.MATH):
            return jvp(field,(x,),(v,))[1]


@contextmanager
def timed(op,key,output):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    old=op.counts.copy()
    yield
    torch.cuda.synchronize()
    log=dict(run=key,wall_s=time.perf_counter()-start,shape=list(op.ids.shape),
             allocated_peak_bytes=torch.cuda.max_memory_allocated(),dtype=str(op.dtype),
             **{k:v-old[k] for k,v in op.counts.items()})
    output.append(log)
    print(key,round(log["wall_s"],3),flush=True)


def rk4(g,x,t,dt):
    a=g(x,t)
    b=g(x+dt*a/2,t+dt/2)
    c=g(x+dt*b/2,t+dt/2)
    d=g(x+dt*c,t+dt)
    return x+dt*(a+2*b+2*c+d)/6


def integrate(g,x,alpha,steps,method="rk4"):
    h=x.float()
    dt=alpha/steps
    for j in range(steps):
        if method=="rk4":
            h=rk4(g,h,j*dt,dt)
        elif method=="midpoint":
            h=h+dt*g(h+dt*g(h,j*dt)/2,(j+.5)*dt)
        else:
            h=h+dt*g(h,j*dt)
    return h


def modified(op,f,alpha,order):
    def field(x,_):
        with sdpa_kernel(SDPBackend.MATH):
            base=f(x).float()
            if order==1:
                return base
            a=op.deriv(f,x,base).float()
            result=base-(alpha/2)*a
            if order==2:
                return result
            eps=.0025
            b=(op.deriv(f,x+eps*base,base)-op.deriv(f,x-eps*base,base))/(2*eps)
            c=op.deriv(f,x,a).float()
            return result+alpha**2*(b/12+c/3)
    return field


def picard(op,f,z,t,iterations=6,damping=1.):
    """Invert x+t*f(x)=z without the true input or output endpoint."""
    h=z
    for _ in range(iterations):
        op.counts["inverse_iterations"]+=1
        h=h+damping*(z-t*f(h).float()-h)
    r=f(h).float()
    return h,r,rel(op,h+t*r,z)


def anderson(op,f,z,t,iterations=6):
    """Residual-minimizing chart inversion with the same evaluations as Picard."""
    h=z
    mapped=[]
    residual=[]
    for _ in range(iterations):
        op.counts["inverse_iterations"]+=1
        next_state=z-t*f(h).float()
        mapped.append(next_state)
        residual.append(((next_state-h)*op.valid[...,None]).flatten(1))
        e=torch.stack(residual,dim=1)
        gram=e@e.transpose(1,2)
        k=gram.shape[1]
        stabilizer=(gram.diagonal(dim1=-2,dim2=-1).mean(-1)*1e-5).clamp_min(1e-12)
        gram=gram+stabilizer[:,None,None]*torch.eye(k,device=z.device)
        weights=torch.linalg.solve_ex(gram,torch.ones_like(gram[:,:,:1]))[0].squeeze(-1)
        weights=weights/weights.sum(-1,keepdim=True)
        h=sum(w[:,None,None]*v for w,v in zip(weights.unbind(1),mapped,strict=True))
    r=f(h).float()
    return h,r,rel(op,h+t*r,z)


def local_record(op,x,y,start,initial):
    return dict(state=rel(op,x,y).tolist(),relative_update_error=(
        norm(op,x-y)/norm(op,y-start).clamp_min(1e-10)).tolist(),
        direction=cosine(op,x-start,y-start).tolist(),
        growth_vs_field=(norm(op,x-start)/norm(op,initial).clamp_min(1e-10)).tolist())


def branch_trial(op,index,alphas):
    x=op.actual[index].float()
    f=lambda h:op.branch(index,h)
    with sdpa_kernel(SDPBackend.MATH):
        v=f(x).float()
        jv=op.deriv(f,x,v)
        native_backend=op.actual[index+1].float()
        out=[]
        for alpha in alphas:
            target=(x+alpha*v)
            tests={}
            for order in (1,2,3):
                g=modified(op,f,alpha,order)
                h=integrate(g,x,alpha,2)
                tests[f"order{order}"]=local_record(op,h,target,x,alpha*v)
            if alpha in (.125,1.):
                g=modified(op,f,alpha,2)
                e=integrate(g,x,alpha,8,method="euler")
                fine=integrate(g,x,alpha,4)
                tests["order2_euler8"]=local_record(op,e,target,x,alpha*v)
                tests["order2_rk4_step_self_error"]=rel(op,fine,integrate(g,x,alpha,2)).tolist()
            root,direction,defect=picard(op,f,target,alpha)
            mid_x=x+alpha*v/2
            _,mid_v,mid_resid=picard(op,f,mid_x,alpha/2)
            tests["chart_midpoint"]=local_record(op,x+alpha*mid_v,target,x,alpha*v)
            out.append(dict(alpha=alpha,methods=tests,inverse_root_error=rel(op,root,x).tolist(),
                            inverse_residual=defect.tolist(),midpoint_inverse_residual=mid_resid.tolist()))
        fd={}
        for eps in (.01,.005,.0025):
            approx=(f(x+eps*v).float()-f(x-eps*v).float())/(2*eps)
            fd[str(eps)]=dict(relative=rel(op,approx,jv).tolist(),cos=cosine(op,approx,jv).tolist())
    return dict(branch=index+1,jr_over_r=(norm(op,jv)/norm(op,v)).tolist(),
        actual_native_discrepancy=rel(op,x+v,native_backend).tolist(),jvp_vs_central=fd,rows=out)


def norm_audit(op):
    module=op.core.norm
    bf_module=deepcopy(module).bfloat16()
    eps=float(module.variance_epsilon)
    h=op.actual[-1].detach()
    scale=module.weight.detach().double()
    ratio_factors=eps/(h.double().square().mean(-1)+eps)
    y64=h.double()*torch.rsqrt(h.double().square().mean(-1,keepdim=True)+eps)*scale

    def recover(y):
        z=y.double()/scale
        discriminant=1-z.square().mean(-1)
        valid=(discriminant>0)&op.valid
        x=z*torch.sqrt(torch.as_tensor(eps,device=z.device)/discriminant.clamp_min(1e-20))[...,None]
        d=(x-h.double()).norm(dim=-1)/h.double().norm(dim=-1)
        return dict(valid=int(valid.sum()),tokens=int(op.valid.sum()),
               error_if_valid=d[valid].tolist(),min_discriminant=float(discriminant[op.valid].min()))
    pairs=[]
    for dose in (1e-5,1e-4,1e-3,.01):
        a=h.float()
        b=a*(1+dose)
        a16,b16=a.bfloat16(),b.bfloat16()
        z16,z_alt16=bf_module(a16),bf_module(b16)
        pairs.append(dict(dose=dose,fp32_input_changed=int(((a!=b).any(-1)&op.valid).sum()),
           fp32_norm_collisions=int(((module(a)==module(b)).all(-1)&op.valid).sum()),
           bf16_input_changed=int(((a16!=b16).any(-1)&op.valid).sum()),
           bf16_norm_collisions_on_distinct_inputs=int(((a16!=b16).any(-1)&(z16==z_alt16).all(-1)&op.valid).sum())))
    return dict(eps=eps,zero_scale=int((scale==0).sum()),negative_scale=int((scale<0).sum()),
        gamma_abs_min=float(scale.abs().min()),gamma_abs_max=float(scale.abs().max()),
        radial_over_tangent=lengths(op,ratio_factors),real_formula=recover(y64),
        actual_fp32=recover(module(h)),output_cast_bf16=recover(module(h).bfloat16()),
        actual_bf16_norm=recover(bf_module(h.bfloat16())),pairs=pairs)


def spectrum(op,field,x,seed,steps=5):
    """Lanczos directions for J^T J. Restricted Ritz values are not extrema certificates."""
    q=noise(op,x,seed)
    q=q/q.norm()
    qs=[]
    ys=[]
    with sdpa_kernel(SDPBackend.MATH):
        _,pull=vjp(field,x)
        for _ in range(steps):
            qs.append(q)
            jq=op.deriv(field,x,q).detach()
            ys.append(jq)
            op.counts["vjp"]+=1
            w=pull(jq)[0].detach()
            for _ in range(2):
                for vec in qs:
                    w=w-(w*vec).sum()*vec
            q=w/w.norm().clamp_min(1e-12)
    mat=torch.stack([a.flatten() for a in ys]).double()
    singular=torch.linalg.eigvalsh(mat@mat.T).clamp_min(0).sqrt().flip(0)
    return dict(singular_values=singular.tolist(),initial_direction_gain=float(ys[0].norm()),
                dimension=x.numel(),probed_dimension=len(qs))
