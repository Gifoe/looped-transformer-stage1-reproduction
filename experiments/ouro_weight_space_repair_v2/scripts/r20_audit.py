"""Read-only first-candidate audit of exact original HF forward kwargs."""
import hashlib
class ForwardAudit:
    def __init__(self,model,scorer,tokens):
        self.module=model.model;self.scorer=scorer;self.tokens=tuple(tokens);self.records=[];self.original=len(self.module._forward_pre_hooks);self.handle=None
        self.backend=model.config._attn_implementation
        assert self.backend=='sdpa'
    def __enter__(self):
        def inspect(module,args,kwargs):
            assert not args and set(kwargs)=={'input_ids','use_cache'} and kwargs['use_cache'] is False
            x=kwargs['input_ids'];actual=tuple(x[0].tolist());assert x.shape==(1,len(self.tokens)) and actual==self.tokens
            self.records.append(dict(kwargs_keys=sorted(kwargs),use_cache=False,input_shape=list(x.shape),input_ids_sha256=hashlib.sha256(str(actual).encode()).hexdigest(),planned_input_exact=True,attention_backend=self.backend,explicit_attention_mask=False,explicit_position_ids=False,past_key_values=False))
            return None
        self.handle=self.module.register_forward_pre_hook(inspect,with_kwargs=True);self.scorer.invalidate();return self
    def __exit__(self,*args):
        self.handle.remove();self.scorer.invalidate();assert len(self.module._forward_pre_hooks)==self.original
