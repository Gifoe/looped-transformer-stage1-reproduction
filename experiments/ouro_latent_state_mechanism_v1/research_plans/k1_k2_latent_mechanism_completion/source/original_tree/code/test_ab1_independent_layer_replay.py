"""Synthetic causal two-loop oracle; no real checkpoint/GPU acceptance."""
import torch
from torch import nn
from types import SimpleNamespace
from ab1_independent_layer_replay import replay, make_boundary, EDGES, bitwise

class Layer(nn.Module):
    attention_type='full_attention'
    def forward(self, x, *, attention_mask, position_ids, past_key_value, use_cache, cache_position, position_embeddings, current_ut):
        assert past_key_value is None and use_cache is False and position_ids.shape==(1, x.shape[1])
        assert attention_mask.shape==(1,1,x.shape[1],x.shape[1]) and current_ut in (0,1)
        return x + torch.cumsum(x, 1)*.002
class Body(nn.Module):
    def __init__(self):
        super().__init__(); self.embed_tokens=nn.Embedding(20,5); self.layers=nn.ModuleList([Layer() for _ in range(24)])
        self.norm=nn.LayerNorm(5); self.early_exit_gate=nn.Linear(5,1); self.total_ut_steps=2
        self.config=SimpleNamespace(num_hidden_layers=24); self.has_sliding_layers=False
    def rotary_emb(self,x,pos): return (pos.float(),pos.float()+1)
class Model(nn.Module):
    def __init__(self):super().__init__();self.model=Body();self.lm_head=nn.Linear(5,20)
def mask_factory(**kw):
    t=kw['input_embeds'].shape[1];return torch.ones(1,1,t,t,dtype=torch.bool).tril()

torch.manual_seed(1114); model=Model().eval();ids=torch.tensor([[1,2,3,4,5,6,7]]);P=2;R=6
with torch.no_grad():
    E=model.model.embed_tokens(ids); x=E.clone()
    for layer in model.model.layers:x=x+torch.cumsum(x,1)*.002
    B=model.model.norm(x); stored={'source_E':E,'source_B':B}
    def oracle(boundary, qsource=None):
        x=boundary.clone(); q9=None
        for i in range(24):
            if i==8:
                q9=x.clone()
                if qsource is not None:x[:,P:R]=qsource[:,P:R]
            x=x+torch.cumsum(x,1)*.002
        return model.lm_head(model.model.norm(x))[:,R,:],q9
    k1=model.lm_head(B)[:,R,:];stored['native_K1_logits']=k1
    N,N9=oracle(B);stored['native_K2_logits']=N;stored['native_Q9_full']=N9
    U,U9=oracle(make_boundary(E,B,P,R,'H01',stored));stored['U_Q9_full']=U9
    names=['H00','H10','H01','H11']+[e+'_'+k for e in EDGES for k in ('norm','random')]+['U_Q9_N','N_Q9_U']
    for name in names:
        if '_norm' in name or '_random' in name:
            edge=name.rsplit('_',1)[0]; receiver,donor,region=EDGES[edge]
            source=make_boundary(E,B,P,R,donor,stored);sl=slice(P,R) if region=='Q' else slice(R,R+1)
            stored[name+'_injected_region']=source[:,sl].clone()
        boundary=make_boundary(E,B,P,R,name,stored)
        qsource=N9 if name=='U_Q9_N' else U9 if name=='N_Q9_U' else None
        stored[name+'_logits'],_=oracle(boundary,qsource)
        result=replay(model,ids,P,R,name,stored,mask_factory=mask_factory)
        assert len(result['layers'])==48
    assert bitwise(oracle(E)[0],k1)
    replay(model,ids,P,R,'E_all',stored,mask_factory=mask_factory)
    stored['H11_logits']=stored['H11_logits']+1
    try:replay(model,ids,P,R,'H11',stored,mask_factory=mask_factory)
    except AssertionError:pass
    else:raise AssertionError('logit mismatch accepted')
print('PASS: 14 endpoints, E-all, H11, 48 explicit calls, cross Q9, tamper refusal; synthetic CPU only')
