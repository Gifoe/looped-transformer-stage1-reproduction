"""Check the adjoints against finite signed writes at the same native receiver."""

import torch

from collect_identification import native, derivatives, pack, parser, readout, reduce_questions, source_rows
from controller_io import controller
import minimal
from task_data import hash_order, save


DOSES=(.01,.05,.10,.50,1.)


def measure(a):
    archive,model,tok,runtime=minimal.runtime(a.archive,a.previous,a.model)
    device=next(model.parameters()).device
    named={key:controller(path,device) for key,path in (arg.split('=',1) for arg in a.operator)}
    records=[]
    max_readout_error=0.
    source=list(source_rows(a.input,a.dataset,a.mode))
    source=sorted(source,key=lambda r:hash_order(r['sample_id'],'loop-identification/finite'))[:(a.limit or 24)]
    for offset in range(0,len(source),a.batch):
        batch=pack(archive,tok,source[offset:offset+a.batch],a.mode)
        h,gd,gl,p,q,ad,al=derivatives(model,batch,a.depth,tok.eos_token_id)
        rows,pos=batch.packed[1:3]
        with torch.no_grad():
            direct,loop,_=native(model,batch,a.depth)
            direct_n=readout(model,direct,batch,tok.eos_token_id)[0]
            loop_n=readout(model,loop,batch,tok.eos_token_id)[0]
            max_readout_error=max(max_readout_error,float((p-direct_n).abs().max()),float((q-loop_n).abs().max()))
            if not torch.allclose(p,direct_n,atol=2e-5,rtol=2e-6) or not torch.allclose(q,loop_n,atol=2e-5,rtol=2e-6):
                raise ValueError('The gradient leaf changes the native unedited prediction')
            cut=native(model,batch,a.depth-1,receiver=a.depth-1)[0]
            if not torch.equal(direct,cut):
                raise ValueError('The direct arm differs from stopping at the same boundary')
            for name,net in named.items():
                write=net.delta(h[rows,pos])
                for dose in DOSES:
                    pair=[]
                    for sign in (1.,-1.):
                        requested=sign*dose*write
                        def edit(x, ii, jj):
                            update=torch.zeros_like(x)
                            return update.index_put((ii,jj),requested)
                        pd,pl,_=native(model,batch,a.depth,edit)
                        u,v=(readout(model,state,batch,tok.eos_token_id)[0] for state in (pd,pl))
                        effective=(h[rows,pos]+requested).to(next(model.parameters()).dtype).float()-h[rows,pos]
                        delta_p=reduce_questions(batch,(gd[rows,pos]*effective).sum(-1))
                        delta_q=reduce_questions(batch,(gl[rows,pos]*effective).sum(-1))
                        pair.append((u-p,v-q,delta_p,delta_q))
                    plus,minus=pair
                    for i,key in enumerate(batch.ids):
                        records.append(dict(id=key,operator=name,dose=dose,
                                            positive=dict(direct=float(plus[0][i]),loop=float(plus[1][i]),
                                                          direct_linear=float(plus[2][i]),loop_linear=float(plus[3][i])),
                                            negative=dict(direct=float(minus[0][i]),loop=float(minus[1][i]),
                                                          direct_linear=float(minus[2][i]),loop_linear=float(minus[3][i])),
                                            native_direct=float(ad[i]),native_loop=float(al[i])))
    a.out.mkdir(parents=True,exist_ok=True)
    save(a.out/'finite.jsonl',records)
    minimal.dump(a.out/'FINITE.json',dict(n=len(source),rows=len(records),depth=a.depth,mode=a.mode,dataset=a.dataset,
                 model_sha256=runtime['model_sha256'],source_sha256=minimal.sha(a.input),doses=DOSES,
                 code_sha256=minimal.sha(__file__),native_stopped_prefix_exact=True,
                 max_gradient_leaf_readout_error=max_readout_error,
                 finite_sha256=minimal.sha(a.out/'finite.jsonl')))
    print('FINITE COMPLETE',a.dataset,a.mode,a.depth,len(source),len(records),flush=True)


if __name__=='__main__':
    measure(parser().parse_args())
