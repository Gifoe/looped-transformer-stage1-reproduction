"""Free greedy decoding under the same prompt-only path interventions."""
import argparse
from pathlib import Path

import torch

from data_protocol import parse_generation
from native import Forward,load,output,pad,save,setup,sha,timer
from up_mediation import PRODUCT_ROLL,capture

MODES=("K1","K2","drop","restore","reverse","product_control")
MAX_NEW=12


class GenerationEdit:
    def __init__(self,lens,drop,product):
        self.lens,self.drop,self.product=lens,drop,product

    def resid(self,k,l,x):
        return x

    def attn(self,key,a,v):
        return a,v

    def mlp(self,key,z,r):
        if key!=(1,0) or self.product is None:
            return z
        y=z.clone()
        for j,n in enumerate(self.lens):
            y[j,:n]=self.product[j]
        return y

    def write(self,key,name,x):
        if key!=(0,0) or name!="a_write" or not self.drop:
            return x
        z=x.clone()
        for j,n in enumerate(self.lens):
            z[j,1:n]=0.
        return z


def encode(tok,prompt):
    txt=prompt.removesuffix("\nAnswer:")+"\nGive only the short answer.\nAnswer:"
    return tok.encode(txt,add_special_tokens=False)


def generate(f,tok,prompts,mode,tensors):
    todo=list(range(len(prompts)))
    results=[[] for _ in prompts]
    task=1 if mode=="K1" else 2
    products=tensors.get(mode)
    cut=mode in ("drop","restore","product_control")
    for _ in range(MAX_NEW):
        ids,pos=pad([prompts[j]+results[j] for j in todo])
        edit=None
        if cut or products is not None:
            edit=GenerationEdit([len(prompts[j]) for j in todo],cut,
                                [products[j] for j in todo] if products is not None else None)
        state=f.run(ids,task,edit=edit)[-1]
        guess=output(f.model,state,pos).argmax(-1).tolist()
        next_todo=[]
        for j,g in zip(todo,guess,strict=True):
            results[j].append(g)
            text=tok.decode(results[j],skip_special_tokens=True)
            if g!=tok.eos_token_id and "\n" not in text.strip():
                next_todo.append(j)
        todo=next_todo
        if not todo:
            break
    return [dict(ids=r,text=tok.decode(r,skip_special_tokens=True)) for r in results]


def block(f,tok,items):
    t=timer()
    p=[encode(tok,q["prompt"]) for q in items]
    U={k:[] for k in ("restore","reverse","product_control")}
    for z in p:
        ids,_=pad([z])
        n=len(z)
        _,c,_=capture(f,ids,n)
        _,d,_=capture(f,ids,n,1.)
        act=f.layers[0].mlp.act_fn
        up_c,up_d=c["u"][0],d["u"][0]
        g_c,g_d=act(c["g_pre"][0]),act(d["g_pre"][0])
        U["restore"].append(up_c*g_d)
        U["reverse"].append(up_d*g_c)
        U["product_control"].append(up_d*g_d+((up_c-up_d)*g_d).roll(PRODUCT_ROLL,-1))
    path={k:generate(f,tok,p,k,U) for k in MODES}
    # Decoding has now ended. Choices are only passed to the scoring function.
    result=[]
    for j,q in enumerate(items):
        row={k:dict(path[k][j],answer=parse_generation(path[k][j]["text"],q["choices"])) for k in MODES}
        result.append(dict(id=q["id"],condition=row,block_seconds=timer()-t if j==0 else 0))
    return result


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("model",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("dataset")
    a=p.parse_args()
    data=load(a.manifest)["cohorts"][a.dataset]["rows"][:128]
    model,tok=setup(a.model,torch.float32)
    f=Forward(model)
    already=load(a.out)["rows"] if a.out.exists() else []
    assert [q["id"] for q in already]==[q["id"] for q in data[:len(already)]]
    for j in range(len(already),len(data),8):
        pack=[{k:v for k,v in q.items() if k!="gold"} for q in data[j:j+8]]
        already.extend(block(f,tok,pack))
        save(a.out,dict(rows=already,manifest_sha256=sha(a.manifest),decode_tokens=MAX_NEW,
                        expected=len(data),peak_bytes=torch.cuda.max_memory_allocated()))
        print(a.dataset,len(already),len(data),flush=True)


if __name__=="__main__":
    main()

