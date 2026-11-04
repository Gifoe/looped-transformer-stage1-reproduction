"""E1 only batches CPU reads of unchanged V zero checks and guard refresh."""
class ValueAblation:
    def __init__(self,model,scorer):
        self.scorer=scorer;self.modules=[layer.self_attn.v_proj for layer in model.model.layers];self.handles=[];self.enabled=False;self.positions=();self.expected_sequence_length=None;self.counts=[0]*len(self.modules)
        self.original=[len(m._forward_hooks) for m in self.modules];self.flags=[];self.verified=True
    def __enter__(self):
        try:
            for i,m in enumerate(self.modules):
                def hook(module,args,output,i=i):
                    self.verified=False;self.counts[i]+=1
                    assert output.shape[1]==self.expected_sequence_length
                    if not self.enabled:return output
                    import torch
                    assert output.dtype==torch.bfloat16 and output.ndim==3 and output.shape[0]==1
                    assert self.positions and max(self.positions)<output.shape[1]
                    result=output.clone();ix=torch.tensor(self.positions,device=output.device);result.index_fill_(1,ix,0)
                    self.flags.append((result.index_select(1,ix)==0).all())
                    return result
                self.handles.append(m.register_forward_hook(hook))
            self.scorer.invalidate();return self
        except BaseException:
            try:self.__exit__(None,None,None)
            except BaseException:pass
            raise
    def configure(self,positions,enabled,expected_sequence_length=None):
        assert self.verified,'Previous forward not verified'
        positions=tuple(positions);assert len(set(positions))==len(positions) and all(i>=0 for i in positions)
        self.expected_sequence_length=expected_sequence_length;self.positions=positions;self.enabled=bool(enabled);self.counts=[0]*len(self.modules);self.flags=[];self.scorer.invalidate()
    def verify(self,loops):
        import torch
        assert self.counts==[loops]*len(self.modules),self.counts
        assert len(self.flags)==(loops*len(self.modules) if self.enabled else 0)
        if self.enabled:assert bool(torch.stack(self.flags).all().item())
        self.verified=True
        return dict(calls_per_physical_layer=list(self.counts),positions=list(self.positions),enabled=self.enabled)
    def __exit__(self,*args):
        error=None
        for h in self.handles:
            try:h.remove()
            except BaseException as e:error=e
        self.handles=[]
        try:
            self.scorer.invalidate();assert [len(m._forward_hooks) for m in self.modules]==self.original
        except BaseException as e:error=e
        if error is not None and not (args and args[0] is not None):raise error
