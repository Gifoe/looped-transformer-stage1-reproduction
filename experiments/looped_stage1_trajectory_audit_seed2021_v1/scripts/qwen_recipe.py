"""Independent implementation of the published block recipe, not author code.

Keeps the pinned HF top-level forward, masks, RoPE, layer indices and final norm.
Only redirects the contiguous window. Prefill body evaluations never write KV;
a separate natural pass writes the canonical `first` KV once. Incremental decode
bypasses recurrence, as requested for log-likelihood evaluation.
"""
from types import MethodType
import torch
import torch.nn.functional as F

class Recipe:
    def __init__(self,model):
        self.model=model;self.inner=model.model
        self.originals={i:self.inner.layers[i].forward for i in [15,16,17,18]}
        self.config=None;self.position=-1;self.record=None

    def restore(self):
        for i,forward in self.originals.items():self.inner.layers[i].forward=forward
        self.config=None

    def set(self,config):
        self.restore()
        if config['K']==0:return
        self.config=config
        controller=self
        def skip(layer,hidden_states,*args,**kwargs):return hidden_states
        def window(layer,hidden_states,*args,**kwargs):
            if args:raise RuntimeError('Unexpected positional Qwen layer inputs')
            return controller.window(hidden_states,kwargs)
        self.inner.layers[15].forward=MethodType(window,self.inner.layers[15])
        for i in [16,17,18]:self.inner.layers[i].forward=MethodType(skip,self.inner.layers[i])

    def window(self,x0,kwargs):
        cache=kwargs.get('past_key_values',kwargs.get('past_key_value'))
        incremental=cache is not None and cache.get_seq_length(15)>0
        def g(x,write=False):
            args=dict(kwargs)
            if not write:
                for key in ['past_key_values','past_key_value']:
                    if key in args:args[key]=None
                args['use_cache']=False
            for i in [15,16,17,18]:
                x=self.originals[i](x,**args)
                if not isinstance(x,torch.Tensor):raise RuntimeError('Pinned Qwen layer return type changed')
            return x
        if incremental:
            # No recurrence during decode; the original cache path executes once.
            return g(x0,write=True)
        cfg=self.config;k=cfg['K'];strategy=cfg['strategy'];x=x0;natural=None;steps=[]
        def change(a,b):
            a=a[0,self.position,:].float();b=b[0,self.position,:].float()
            return dict(relative_l2=((b-a).norm()/a.norm().clamp_min(1e-8)).item(),cosine=F.cosine_similarity(a,b,dim=0).item())
        for i in range(k):
            y=g(x)
            if i==0:natural=y
            # Preserve the mathematical K=1 identity without BF16 subtraction
            # cancellation. This does not alter any fixed K=2/6 experiment.
            new=y if strategy=='naive' or k==1 else x+(y-x)/k
            steps.append(dict(iteration=i+1,**change(x,new)));x=new
        if strategy=='block_anchored':
            beta=cfg['anchor_beta']
            x=natural if beta==1 else beta*natural+(1-beta)*x
        if cache is not None:
            # Author README and paper cache=first: write natural pre-loop KV;
            # discard this pass's hidden output and keep the chosen loop output.
            g(x0,write=True)
        self.record=dict(input_to_output=change(x0,x),input_to_natural=change(x0,natural),steps=steps,
                         cache_strategy='first',decode_mode='bypass')
        return x
