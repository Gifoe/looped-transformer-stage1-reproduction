"""Collect true state-intervention targets with unchanged frozen scorer functions."""
import sys,json,time,hashlib,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEV=ROOT.parent/'closed_loop_action_value_dev400_20261029'
OLD=ROOT.parent/'frozen_loop_update_value_seed2021_v1'
sys.path.insert(0,str(DEV/'scripts'))
import run_development_packed as core
import torch
import numpy as np
from transformers import AutoTokenizer,AutoModelForCausalLM
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):
    p=ROOT/'results'/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2),encoding='utf8');tmp.replace(p)
def main():
    split=sys.argv[1];assert split in ['head_train','validation'];begin=time.monotonic()
    lock=json.loads((ROOT/'configs/LOCK.json').read_text())
    for rel,h in lock['files'].items():assert sha(ROOT/rel)==h,rel
    for rel,h in lock['dependencies'].items():assert sha(ROOT.parent/rel)==h,rel
    if split=='validation':
        train_lock=json.loads((ROOT/'configs/TRAIN_LOCK.json').read_text())
        assert sha(ROOT/'configs/heads.json')==train_lock['heads_sha256']
    path=OLD/'data'/f'{split}.jsonl'
    with path.open(encoding='utf8') as f:rows=[json.loads(l) for l in f]
    assert [r['id'] for r in rows]==lock['manifests'][split]['ids']
    with (OLD/'results'/f'{split}.jsonl').open(encoding='utf8') as f:reference={r['id']:r['views']['official']['sum_scores']['adaptive'] for r in map(json.loads,f)}
    torch.set_num_threads(4);torch.set_grad_enabled(False);torch.manual_seed(2021);torch.backends.cuda.matmul.allow_tf32=False
    directory=ROOT.parent/'looped_stage1_trajectory_audit_seed2021_v1/models/Ouro-1.4B'
    assert json.loads((directory/'snapshot_metadata.json').read_text())['sha']=='574fa66cb8bf5abdc979642d01cf2b79b16bfab1'
    tok=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    model,info=AutoModelForCausalLM.from_pretrained(directory,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',output_loading_info=True)
    assert not any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs'])
    model=model.to('cuda').eval();model.requires_grad_(False);versions={n:p._version for n,p in model.named_parameters()}
    assert model.model.total_ut_steps==4
    torch.cuda.reset_peak_memory_stats();maxerr=0.;row_times=[];outpath=ROOT/'results'/f'{split}.jsonl'
    with outpath.open('x',encoding='utf8') as out:
        for i,row in enumerate(rows):
            assert time.time()+30<float(os.environ['CALIBRATION_DEADLINE']),'Budget reserve'
            if __import__('shutil').disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Disk reserve')
            start=time.monotonic();pairs=row['variants']['official'];assert len({p[0] for p in pairs})==1
            decision,controls,decision_s=core.decision_with_controls(tok,model,pairs[0][0],row['question'])
            actions={};traces={};terminal=None
            for a in core.ALPHAS:
                scores,term,effects,trace,timing=core.evaluate(tok,model,pairs,a,row['question']);traces[a]=trace
                actions[str(a)]=dict(sum_scores=scores,prediction=int(np.argmax(np.array(scores)/row['choice_char_lengths'])),intervention_rms=effects,**timing)
                if a==0:terminal={k:dict(sum_scores=v,prediction=int(np.argmax(np.array(v)/row['choice_char_lengths']))) for k,v in term.items()}
            drift={str(a):[float((traces[a][j]-traces[0.][j]).square().mean().sqrt()) for j in range(4)] for a in core.ALPHAS[1:]}
            assert all(v[0]==0 and min(v[1:])>0 for v in drift.values())
            error=max(abs(a-b) for a,b in zip(actions['0.0']['sum_scores'],reference[row['id']]))
            assert error<1e-5,error;maxerr=max(maxerr,error)
            rec=dict(id=row['id'],decision=decision,controls=controls,actions=actions,terminal=terminal,downstream_state_rms=drift,baseline_score_error=error,decision_seconds=decision_s,choice_char_lengths=row['choice_char_lengths'],gold=row['gold'],elapsed_s=time.monotonic()-start)
            out.write(json.dumps(rec,ensure_ascii=False)+'\n');out.flush();row_times.append(rec['elapsed_s'])
            if split=='head_train' and i==15:
                seconds_per_question=float(np.mean(row_times));remaining_questions=len(rows)-16+800
                projection=1.1*seconds_per_question*remaining_questions+60
                remaining=float(os.environ['CALIBRATION_DEADLINE'])-time.time()
                save('THROUGHPUT.json',dict(n=16,seconds_per_question=seconds_per_question,projected_remaining_seconds_with_reserve=projection,remaining_seconds=remaining,within_budget=projection<remaining))
                assert projection<remaining,'Measured throughput exceeds bounded collection budget'
            if (i+1)%50==0 or i==15:
                status=dict(split=split,completed=i+1,expected=len(rows),elapsed_s=time.monotonic()-begin,pid=os.getpid())
                save('STATUS.json',status);print(json.dumps(status),flush=True)
    assert all(p._version==versions[n] and not p.requires_grad for n,p in model.named_parameters())
    save(split+'_COMPLETE.json',dict(n=len(rows),sha256=sha(outpath),baseline_max_score_error=maxerr,weights_unchanged=True,all_nonzero_actions_propagated=True,elapsed_s=time.monotonic()-begin,peak_vram_bytes=torch.cuda.max_memory_allocated(),source_lock_sha256=sha(ROOT/'configs/LOCK.json')))
if __name__=='__main__':main()
