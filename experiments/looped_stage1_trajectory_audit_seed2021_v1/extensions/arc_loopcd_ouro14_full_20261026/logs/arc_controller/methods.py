"""Source-audited training-free scoring kernels; no trainable parameters."""
import ast
from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from common import ROOT,CFG

def adaptive_logits(final,early,cap):
    top=final.float().softmax(-1).topk(2,dim=-1).values
    strength=float(cap)*(1-(top[...,0]-top[...,1]))
    return final.float()+strength[...,None]*(final.float()-early.float()),strength

def projection_controls(final,early,strength):
    final=final.float();delta=final-early.float()
    centered=final-final.mean(-1,keepdim=True);delta=delta-delta.mean(-1,keepdim=True)
    coefficient=(delta*centered).sum(-1,keepdim=True)/centered.square().sum(-1,keepdim=True).clamp_min(1e-20)
    parallel=coefficient*centered;orthogonal=delta-parallel
    return centered+strength*parallel,centered+strength*orthogonal,parallel,orthogonal

def dola_scores(final,premature,layer_ids):
    # Deliberately replicate the official code's KL argument order rather than
    # silently replacing it with the textbook Jensen-Shannon expression.
    final=final.float();premature=premature.float()
    pf=final.softmax(-1);pp=premature.softmax(-1)
    mixture=.5*(pf[None]+pp)
    lf=final.log_softmax(-1);lp=premature.log_softmax(-1)
    divergence=.5*(F.kl_div(lf[None],mixture,reduction='none').mean(-1)+F.kl_div(lp,mixture,reduction='none').mean(-1))
    chosen=divergence.argmax(0)
    early=lp[chosen,torch.arange(final.shape[0],device=final.device)]
    contrasted=(lf-early).log_softmax(-1)
    threshold=lf.max(-1,keepdim=True).values+np.log(CFG['dola']['relative_top'])
    mask=lf<threshold
    # Official lm_score normalizes BEFORE masking and does not normalize again.
    output=torch.where(mask,CFG['dola']['masked_value'],contrasted)
    selected=[layer_ids[int(i)] for i in chosen]
    return output,selected

@contextmanager
def qwen_premature(model,selection,layers):
    captured={};handles=[]
    for index in layers:
        def capture(module,args,index=index):captured[index]=args[0][:,selection,:].clone()
        handles.append(model.model.layers[index].register_forward_pre_hook(capture))
    try:yield captured
    finally:
        for handle in handles:handle.remove()

def official_dola_oracle(final,premature,layer_ids,targets):
    """Execute the unmodified official lm_score AST on supplied real logits."""
    source=ast.parse((ROOT/'sources/dola.py').read_text(encoding='utf-8'))
    cls=next(node for node in source.body if isinstance(node,ast.ClassDef) and node.name=='DoLa')
    functions=[node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in ['get_relative_top_filter','lm_score']]
    container=ast.Module(body=[ast.ClassDef(name='ReferenceDoLa',bases=[],keywords=[],body=functions,decorator_list=[])],type_ignores=[])
    namespace=dict(torch=torch,F=F,np=np)
    exec(compile(ast.fix_missing_locations(container),str(ROOT/'sources/dola.py'),'exec'),namespace)
    reference=namespace['ReferenceDoLa']();reference.device=str(final.device)
    t=len(targets);n=3+t;mapping={}
    for layer,z in [(CFG['dola']['mature_layer'],final)]+[(layer,premature[i]) for i,layer in enumerate(layer_ids)]:
        padded=torch.zeros((1,n,z.shape[-1]),device=z.device,dtype=z.dtype)
        padded[0,2:2+t]=z;mapping[layer]=padded
    prefix=torch.tensor([[1,2,3]],device=final.device)
    full=torch.cat([prefix,targets[None]],dim=1)
    reference.tokenizer=lambda text,**kw:SimpleNamespace(input_ids=prefix if text=='prefix' else full)
    reference.model=lambda **kw:(mapping,None)
    return reference.lm_score('prefix',' continuation',mode='dola',mature_layer=CFG['dola']['mature_layer'],candidate_premature_layers=layer_ids,relative_top=CFG['dola']['relative_top'],post_softmax=True)

def native_cdf(gates):
    remaining=torch.ones_like(gates[0][0,-1,0]);pdf=[]
    for index,gate in enumerate(gates):
        value=torch.sigmoid(gate[0,-1,0])
        if index<len(gates)-1:pdf.append(value*remaining);remaining=remaining*(1-value)
        else:pdf.append(remaining)
    return torch.stack(pdf).cumsum(0)
