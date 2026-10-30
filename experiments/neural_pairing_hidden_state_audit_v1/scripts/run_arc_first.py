"""Complete ARC-Challenge ON/CUT first; both receivers, exact official 4-choice filter.

This is final nl33w, not missing QA-trained nl23. No generation or MMLU prerequisite.
"""
import argparse
import hashlib
import json
import time
from linux_runtime import ROOT, load_frozen, save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--bs',type=int,default=16);ap.add_argument('--seconds',type=int,default=3600)
    args=ap.parse_args();assert args.bs>=2
    import pyarrow.parquet as pq
    source=ROOT/'sources/arc_challenge_test_pinned.parquet'
    assert hashlib.sha256(source.read_bytes()).hexdigest()=='62f03257e737aed263f55c6abf87c7bb0028a44a6bdd2a26eb1279eb42c1d1e9'
    raw=pq.read_table(source).to_pylist();items=[];original_ids=[]
    for row in raw:
        labels=row['choices']['label']
        if len(labels)!=4 or row['answerKey'] not in labels:continue
        items.append(dict(src='arc_challenge',q=row['question'],choices=row['choices']['text'],answer=labels.index(row['answerKey'])))
        original_ids.append(row['id'])
    assert len(raw)==1172 and len(items)==1165
    published=json.loads((ROOT/'sources/published/results/magic-nl23/bench.json').read_text())
    expected=[r for r in published['questions'] if r['src']=='arc_challenge']
    assert [dict(src=it['src'],answer=it['answer']) for it in items]==expected
    out=ROOT/'fresh/arc_first';out.mkdir(exist_ok=True)
    config=dict(evidence_type='fresh_final_checkpoint_inference',checkpoint='final-nl33w',
        checkpoint_revision='29e7ea33878c5bdc2c3cf4e27fffe46b4966859c',dataset_revision='210d026faf9955653af8916fad021475a3f00453',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),raw_test_n=1172,evaluated_n=1165,
        filtering='Official: exactly four choices and valid answer key; seven other-choice-count items excluded',
        bs=args.bs,conditions=['link','cut'],prompt_style='official pair',
        weights='Same released adapters and bridges in ON/CUT; no QA Solo or Text-Pair substitutions',
        note='First complete benchmark is prioritized. Full mechanism audit and other controls follow separately.')
    if (out/'config.json').exists():assert json.loads((out/'config.json').read_text())==config
    else:save(out/'config.json',config)
    save(out/'items.json',dict(items=items,original_ids=original_ids))
    start=time.time()
    save(out/'WORKER.json',dict(status='loading',started_unix=start))
    pair,toks=load_frozen()
    import torch
    import games as G
    import session as S
    import bench_eval
    from mcq_tools import prepare,letter_logits,outcomes
    torch.set_num_threads(4)
    lengths=[]
    for it in items:
        b=S.build(G.qa_episode(it,'pair'),toks)
        lengths.append(max(len(b[x]['ids']) for x in b))
    rank=sorted(range(len(items)),key=lambda i:lengths[i])
    batches=list(S.batches(len(items),args.bs))
    with torch.inference_mode():
        for bid,(a,z) in enumerate(batches):
            ids=rank[a:z];chunk=[items[i] for i in ids]
            for cond,mode in [('link',None),('cut','zero')]:
                target=out/f'batch_{bid:04d}_{cond}.json'
                if target.exists():continue
                if time.time()-start > args.seconds:
                    save(out/'WORKER.json',dict(status='time_limit',elapsed_seconds=time.time()-start,
                        completed_rank_questions=a,total_questions=len(items),gpu_worker_seconds=time.time()-start))
                    return
                begin=time.time();torch.cuda.reset_peak_memory_stats()
                b,pos,_=prepare(pair,chunk)
                last,_=pair.joint(b,mode=mode)
                lg=letter_logits(pair,last,pos)
                if bid==0:
                    expected=bench_eval.letter_probs(pair,chunk,'pair',mode=mode,bs=len(chunk))
                    for x in pair.models:
                        torch.testing.assert_close(lg[x].softmax(-1),expected[x].to(lg[x].device),atol=1e-6,rtol=1e-5)
                result=outcomes(lg,chunk)
                torch.cuda.synchronize()
                save(target,dict(evidence_type='fresh_final_checkpoint_inference',condition=cond,
                    question_indices=ids,original_ids=[original_ids[i] for i in ids],per_receiver=result,
                    elapsed_seconds=time.time()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                del last,lg
            elapsed=time.time()-start
            save(out/'PROGRESS.json',dict(status='running',completed_questions=z,total_questions=len(items),
                completed_batches=bid+1,total_batches=len(batches),elapsed_seconds=elapsed,
                elapsed_seconds_per_question_two_conditions=elapsed/z,
                estimated_remaining_seconds=elapsed/z*(len(items)-z),updated_unix=time.time()))
            print(json.dumps(dict(completed_questions=z,total=len(items),elapsed_seconds=elapsed)),flush=True)
    receipt=dict(status='complete',n=len(items),finished_unix=time.time(),gpu_worker_seconds=time.time()-start,config=config)
    save(out/'COMPLETE.json',receipt);save(out/'WORKER.json',receipt)


if __name__=='__main__':main()
