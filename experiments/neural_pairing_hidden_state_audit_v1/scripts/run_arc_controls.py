"""Additional ARC controls, reusing the completed ON/CUT questions and exact shuffle batches."""
import argparse
import contextlib
import json
import time
from linux_runtime import ROOT,load_frozen,save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seconds',type=int,default=3600);args=ap.parse_args()
    out=ROOT/'fresh/arc_first'
    assert json.loads((out/'COMPLETE.json').read_text())['status']=='complete'
    data=json.loads((out/'items.json').read_text());items=data['items']
    pair,toks=load_frozen()
    import torch
    import bench_eval
    from mcq_tools import prepare,letter_logits,outcomes
    start=time.time()
    for bid,path in enumerate(sorted(out.glob('batch_*_cut.json'))):
        old=json.loads(path.read_text());ids=old['question_indices'];chunk=[items[i] for i in ids]
        assert len(chunk)>=2
        for name,mode,style in [('base','zero','solo'),('shuffle','shuffle','pair'),('random','random','pair')]:
            target=out/(path.name.removesuffix('cut.json')+name+'.json')
            if target.exists():continue
            if time.time()-start>args.seconds:
                save(out/'CONTROLS_PROGRESS.json',dict(status='time_limit',batch=bid,gpu_worker_seconds=time.time()-start));return
            begin=time.time();torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode(),contextlib.ExitStack() as stack:
                if name=='base':
                    for model in pair.models.values():stack.enter_context(model.disable_adapter())
                b,pos,_=prepare(pair,chunk,style=style)
                last,_=pair.joint(b,mode=mode,gen=torch.Generator().manual_seed(2021+bid))
                lg=letter_logits(pair,last,pos)
                if bid==0 and name=='base':
                    expected=bench_eval.letter_probs(pair,chunk,'solo',bs=len(chunk))
                    for x in pair.models:torch.testing.assert_close(lg[x].softmax(-1),expected[x].to(lg[x].device),atol=1e-6,rtol=1e-5)
                result=outcomes(lg,chunk)
                torch.cuda.synchronize()
            save(target,dict(evidence_type='fresh_final_checkpoint_inference',condition=name,question_indices=ids,
                original_ids=old['original_ids'],per_receiver=result,elapsed_seconds=time.time()-begin,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),random_seed=2021+bid,
                shuffle_donors={x:[ids[(j-shift)%len(ids)] for j in range(len(ids))] for x,shift in [('quill',1),('ember',-1)]} if name=='shuffle' else None))
            del last,lg
        save(out/'CONTROLS_PROGRESS.json',dict(status='running',completed_batches=bid+1,total_batches=73,updated_unix=time.time(),elapsed_seconds=time.time()-start))
        print(json.dumps(dict(completed_batches=bid+1)),flush=True)
    save(out/'CONTROLS_COMPLETE.json',dict(status='complete',n=len(items),conditions=['base','shuffle','random'],gpu_worker_seconds=time.time()-start,finished_unix=time.time()))


if __name__=='__main__':main()
