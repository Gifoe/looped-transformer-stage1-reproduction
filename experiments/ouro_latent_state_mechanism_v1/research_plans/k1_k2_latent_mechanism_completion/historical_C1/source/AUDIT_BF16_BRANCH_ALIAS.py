"""CPU-only synthetic hook/alias check, not model or scientific validation."""
import hashlib
import json
from pathlib import Path
import torch

class Tail(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_mlp = torch.nn.Identity()
        self.mlp = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
    def forward(self, x, a):
        y = x + a
        residual = y
        y = self.pre_mlp(y)
        y = self.mlp(y)
        return residual + y

torch.manual_seed(67104)
tail = Tail().eval()
shape = (1, 5, 16)
xu, xn, au, an = [torch.randn(shape).to(torch.bfloat16) for _ in range(4)]
cases = {}
with torch.no_grad():
    for name, x, a in [('U',xu,au),('restore_x',xn,au),('restore_a',xu,an),('joint',xn,an)]:
        target = xu + au
        target[:, -1] = (x+a)[:, -1]
        expected = target + tail.mlp(target)
        seen = {}
        def inplace_hook(module, args):
            value = args[0]
            before = value.clone()
            value[:, -1] = target[:, -1]
            assert torch.equal(value[:, :-1], before[:, :-1])
            seen['object_id'] = id(value)
            seen['storage_ptr'] = value.untyped_storage().data_ptr()
        handle = tail.pre_mlp.register_forward_pre_hook(inplace_hook)
        actual = tail(xu, au)
        handle.remove()
        assert torch.equal(actual, expected), name
        def clone_hook(module, args):
            value = args[0].clone()
            value[:, -1] = target[:, -1]
            return (value,)
        handle = tail.pre_mlp.register_forward_pre_hook(clone_hook)
        clone_actual = tail(xu, au)
        handle.remove()
        different = int((clone_actual != expected).sum())
        if name != 'U':
            assert different > 0, name
        cases[name] = {'inplace_exact_explicit_branch_sum':True,'clone_return_wrong_elements':different}
out = {'status':'PASS','device':'cpu','dtype':'bfloat16','cases':cases,
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'torch':torch.__version__,
       'scope':'Synthetic arithmetic/alias illustration only. Does not certify pinned Ouro layer, SDPA, full suffix or any scientific effect. Real technical endpoint acceptance is still required.'}
path = Path(__file__).with_name('BF16_BRANCH_ALIAS_CPU_AUDIT.json')
path.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
