"""Instrumentation and interventions on the pinned native, eager, no-cache Ouro forward."""
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import import_utils

# The text-only runtime does not require the server image torchvision build.
import_utils._torchvision_available=False

SEED=20261121
REV="574fa66cb8bf5abdc979642d01cf2b79b16bfab1"
MODEL_HASH="58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af"


def sha(path):
    with open(path,"rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path,obj):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)


def setup(model_path,dtype=torch.float32):
    torch.manual_seed(SEED)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True,
              torch_dtype=torch.bfloat16,attn_implementation="eager",local_files_only=True).cuda().eval()
    model.requires_grad_(False).to(dtype)
    tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True,local_files_only=True)
    assert len(model.model.layers)==24 and model.model.config._attn_implementation=="eager"
    assert not torch.is_autocast_enabled("cuda")
    return model,tokenizer


def tensors(tok,texts):
    ids=[tok.encode(t,add_special_tokens=False) for t in texts]
    return pad(ids)


def pad(ids):
    n=max(map(len,ids))
    a=torch.tensor([x+[0]*(n-len(x)) for x in ids],device="cuda")
    positions=torch.tensor([len(x)-1 for x in ids],device="cuda")
    return a,positions


class Forward:
    def __init__(self,model):
        self.model=model
        self.core=model.model
        self.module=sys.modules[type(self.core.layers[0].self_attn).__module__]
        self.layers=self.core.layers

    def run(self,ids,k,*,start=None,edit=None,record=None):
        self.core.total_ut_steps=k
        params=dict(use_cache=False)
        params["input_ids" if start is None else "inputs_embeds"]=ids if start is None else start
        options=contextlib.nullcontext() if record is None and edit is None else self.observe(record,edit)
        with options:
            _,states,_=self.core(**params)
        return states

    @contextlib.contextmanager
    def observe(self,record=None,edit=None):
        record=record if record is not None else Recorder(keep=False)
        handles=[]
        active=None
        native=self.module.eager_attention_forward

        def prefix(layer,args,kwargs,index):
            nonlocal active
            h=args[0]
            k=kwargs["current_ut"]
            active=(k,index)
            assert not kwargs["use_cache"] and kwargs["past_key_value"] is None
            record.put(active,"input",h)
            if edit is not None:
                y=edit.resid(k,index,h)
                if y is not h:
                    return ((y,*args[1:]),kwargs)
            return None

        def capture(_,args,output,name):
            record.put(active,name,output)

        def hook(value,name):
            def apply(_,args,output):
                capture(_,args,output,name)
                if edit is not None and name in ("a_write","m_write") and hasattr(edit,"write"):
                    return edit.write(active,name,output)
            handles.append(value.register_forward_hook(apply))

        def eager(module,q,k,v,mask,scaling,dropout=0.,**kw):
            if edit is not None and hasattr(edit,"qk"):
                q,k=edit.qk(active,q,k)
            att,a=native(module,q,k,v,mask,scaling,dropout=dropout,**kw)
            record.put(active,"q",q)
            record.put(active,"k",k)
            if edit is not None:
                a2,v2=edit.attn(active,a,v)
                if a2 is not a or v2 is not v:
                    att=(a2 @ self.module.repeat_kv(v2,module.num_key_value_groups)).transpose(1,2).contiguous()
                    a,v=a2,v2
            record.put(active,"v",v)
            record.put(active,"a",a)
            record.put(active,"av",att)
            return att,a

        def down(_,args):
            record.put(active,"z",args[0])
            if edit is not None:
                z=edit.mlp(active,args[0],record)
                if z is not args[0]:
                    return (z,)
            return None

        try:
            self.module.eager_attention_forward=eager
            for i,l in enumerate(self.layers):
                handles.append(l.register_forward_pre_hook(lambda layer,a,k,i=i:prefix(layer,a,k,i),with_kwargs=True))
                for sub,name in ((l.self_attn.o_proj,"att_projected"),(l.input_layernorm_2,"a_write"),
                     (l.post_attention_layernorm,"mlp_input"),(l.mlp.gate_proj,"g_pre"),
                     (l.mlp.up_proj,"u"),(l.mlp.down_proj,"down"),(l.post_attention_layernorm_2,"m_write"),(l,"output")):
                    hook(sub,name)
                handles.append(l.mlp.down_proj.register_forward_pre_hook(down))
            yield
        finally:
            self.module.eager_attention_forward=native
            for handle in handles:
                handle.remove()


class Recorder:
    def __init__(self,keep=True,keys=None):
        self.data={}
        self.keep=keep
        self.keys=keys

    def put(self,key,name,x):
        if self.keep and (self.keys is None or key in self.keys):
            self.data.setdefault(key,{})[name]=x.detach()


def output(model,h,positions):
    return model.lm_head(h[torch.arange(len(h),device=h.device),positions]).float()


def flatnorm(a):
    return a.float().flatten(1).norm(dim=1)


def effect_metrics(model,d,a,b,positions):
    da=(a-d).float()
    db=(b-d).float()
    relative=(flatnorm(da-db)/flatnorm(db).clamp_min(1e-9))
    cosine=(da.flatten(1)*db.flatten(1)).sum(-1)/(flatnorm(da)*flatnorm(db)).clamp_min(1e-9)
    return dict(to_native_relative=relative.tolist(),update_cosine=cosine.tolist(),
                token_error=(a-b).float().norm(dim=-1).tolist(),
                readout_magnitude=(output(model,a,positions)-output(model,d,positions)).norm(dim=-1).tolist())


def timer():
    torch.cuda.synchronize()
    return time.monotonic()


def identity_trial(f,ids,positions):
    r=Recorder()
    clean=f.run(ids,4)
    traced=f.run(ids,4,record=r)
    assert all(torch.equal(a,b) and torch.equal(output(f.model,a,positions),output(f.model,b,positions))
               for a,b in zip(clean,traced,strict=True))
    core=f.core
    h=core.embed_tokens(ids)
    base=f.run(ids,1,start=h)[0]
    class SameInput:
        def resid(self,k,l,x):
            return h if l==0 else x
        def attn(self,_,a,v):
            return a,v
        def mlp(self,_,z,r):
            return z
    rest=f.run(ids,4,start=h,edit=SameInput())
    assert all(torch.equal(base,z) for z in rest)
    assert all(torch.equal(r.data[k,l]["input"] + r.data[k,l]["a_write"] + r.data[k,l]["m_write"],
                           r.data[k,l]["output"]) for k in range(4) for l in range(24))
    return dict(loop_outputs_bitwise=True,operator_same_input_loop_0_to_3_bitwise=True,
                native_residual_boundaries=192, vocabulary_logits=4*len(ids)*f.core.config.vocab_size)


def dump_witness(out,ids,states,r):
    for k in range(4):
        for part in range(0,24,6):
            data={f"l{li+1}_{name}":z.cpu().numpy() for (ki,li),row in r.data.items()
                  if ki==k and part<=li<part+6 for name,z in row.items()}
            data["token_ids"]=ids.cpu().numpy()
            data["final_norm"]=states[k].cpu().numpy()
            np.savez_compressed(out/f"TRACE_K{k+1}_L{part+1}.npz",**data)

