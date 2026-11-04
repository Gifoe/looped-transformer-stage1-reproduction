"""CPU synthetic parity, hook cleanup, no dataset or model loading."""
import json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from r20_math import synthetic_checks,numpy_controls,torch_controls
from r20_admission import save,sha

def main():
    result=synthetic_checks();rng=np.random.default_rng(2021);a=[rng.normal(size=(3,19)).astype(np.float32) for _ in range(6)];perm=rng.permutation(19)
    n,nd=numpy_controls(*a,perm);t,td=torch_controls(*(torch.from_numpy(x) for x in a),perm)
    error=max(float(np.max(np.abs(n[k]-t[k].numpy()))) for k in n);assert error<=2e-5
    assert np.max(np.abs(nd['alpha']-td['alpha'].numpy()))<=2**-32
    from types import SimpleNamespace
    from r20_hooks import ValueAblation
    modules=[torch.nn.Identity(),torch.nn.Identity()];model=SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(self_attn=SimpleNamespace(v_proj=m)) for m in modules]))
    class Scorer:
        calls=0
        def invalidate(self):self.calls+=1
    scorer=Scorer();x=torch.ones((1,5,3),dtype=torch.bfloat16)
    with ValueAblation(model,scorer) as hook:
        hook.configure([1,2],False,5);assert all(m(x) is x for m in modules);hook.verify(1)
        hook.configure([1,2],True,5)
        for _ in range(4):
            for m in modules:
                y=m(x);assert torch.equal(y[:,[0,3,4]],x[:,[0,3,4]]) and bool((y[:,[1,2]]==0).all()) and bool((x==1).all())
        hook.verify(4)
    assert all(not m._forward_hooks for m in modules)
    result.update(status='passed_CPU_numpy_torch_hook',Torch_parity='passed_CPU',Torch_parity_logit_error=error,hook_off_tensor_identity=True,hook_active_only_selected_positions=True,hook_finally_cleanup=True,source_sha256={p:sha('scripts/'+p) for p in ['r20_math.py','r20_hooks.py','r20_cpu_checks.py']},scope='CPU synthetic only; no model, data, GPU worker or heldout')
    save('results/r20_CPU_MATH_CHECKS.json',result);print(json.dumps(result))
if __name__=='__main__':main()

