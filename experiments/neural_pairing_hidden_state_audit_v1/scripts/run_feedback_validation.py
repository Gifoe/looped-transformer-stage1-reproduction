"""Exhaust remaining correct-peer harm events; original batches, no training."""
import argparse
import json
import time
from linux_runtime import ROOT, load_frozen, save


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',default='feedback_validation');args=ap.parse_args()
    out=ROOT/'fresh'/args.study;out.mkdir(exist_ok=True)
    plan=json.loads((ROOT/'configs/feedback_validation.json').read_text())
    save(out/'SELECTION.json',plan)
    start=time.time();pair,toks=load_frozen()
    import torch
    from causal_trace import CausalTrace
    from mcq_tools import prepare,letter_logits,outcomes,frozen_readout_rows,add_dot64_readout
    readouts=frozen_readout_rows(pair)
    with torch.inference_mode():
        completed=0
        for bid,group in enumerate(plan['batches']):
            path=out/f'batch_{bid:03d}.json'
            if path.exists():completed+=len(group['selected_ids']);continue
            cases=group['cases'];items=[c['item'] for c in cases]
            b,own,built=prepare(pair,items);positions=own
            rec={};traces={};timing={}
            for name,kw in [('cut',dict(keep_receivers=[])),('link',{})]:
                before=time.time()
                with CausalTrace(pair,positions,capture_full=True,**kw) as tr:last,_=pair.joint(b)
                rec[name]=outcomes(letter_logits(pair,last,own),items);add_dot64_readout(rec[name],last,own,readouts)
                traces[name]=tr;timing[name]=time.time()-before
                targets=[i for i,c in enumerate(cases) if c['selected']]
                torch.save(dict(target_rows=targets,states={k:v[targets] for k,v in tr.states.items()}),out/f'batch_{bid:03d}_{name}_selected_states.pt')
                for i,c in enumerate(cases):
                    for x in pair.models:
                        assert rec[name][x][i]['letter_logits']==c['saved_baseline'][name][x]['letter_logits'],('Original replay mismatch',c['question_id'],name,x)
            receivers=sorted({c['target_receiver'] for c in cases if c['selected']})
            specs=[]
            for x in receivers:
                y='ember' if x=='quill' else 'quill'
                donors={(y,l):traces['cut'].full_pre[(y,l)] for l in pair.xblocks}
                norms={(x,l):traces['link'].full_write_norms[(x,l)] for l in pair.xblocks}
                specs += [(f'oneway_to_{x}',dict(keep_receivers=[x])),
                    (f'clean_kv_from_{y}',dict(sender_kv_donors=donors)),
                    (f'clean_gate_from_{y}',dict(sender_gate_donors=donors)),
                    (f'clean_kv_normmatched_from_{y}',dict(sender_kv_donors=donors,write_norms=norms))]
            specs += [('link_repeat',{})]
            for name,kw in specs:
                before=time.time()
                with CausalTrace(pair,positions,capture=False,**kw):last,_=pair.joint(b)
                rec[name]=outcomes(letter_logits(pair,last,own),items);add_dot64_readout(rec[name],last,own,readouts)
                timing[name]=time.time()-before
                save(out/'ACTIVE_CONDITION.json',dict(batch=bid,condition=name,elapsed_seconds=time.time()-start))
            assert rec['link']==rec['link_repeat'],'Same-shape repeat changed'
            save(path,dict(cases=cases,conditions=rec,timing_seconds=timing,original_batch=group['original_batch'],
                checks=dict(original_baseline_bitwise=True,repeat_identical=True),
                interpretation='Exhaustive remaining correct-peer harm target events after excluding all previously traced questions; neighboring rows are padding controls. All scores retain original row order and ticks.'))
            completed+=len(group['selected_ids']);del traces
            save(out/'PROGRESS.json',dict(completed_targets=completed,total_targets=plan['n_targets'],completed_batches=bid+1,total_batches=len(plan['batches']),elapsed_seconds=time.time()-start))
            print(json.dumps(dict(completed_targets=completed,elapsed_seconds=time.time()-start)),flush=True)
    save(out/'COMPLETE.json',dict(status='complete',n_targets=plan['n_targets'],gpu_worker_seconds=time.time()-start,finished_unix=time.time()))


if __name__=='__main__':main()
