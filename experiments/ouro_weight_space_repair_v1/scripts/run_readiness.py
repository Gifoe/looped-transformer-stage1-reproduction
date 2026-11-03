"""Unique real-checkpoint gate: native depth and cache parity; no test inputs."""
import argparse,json,time
from runtime import ROOT,load_model,encode_pair,save,check_budget
from ouro_cached_scoring import OuroCachedScorer

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--native-only',action='store_true');args=ap.parse_args()
 start=time.time();tok,model=load_model();scorer=OuroCachedScorer(model,6)
 import torch
 text='The Earth orbits the Sun. The answer is'
 pairs=[encode_pair(tok,text,a) for a in [' yes',' no',' a star']]
 depths=scorer.check_depth_invariance(pairs,depths=(3,4,5),atol=0.0);assert depths['passed'],depths
 if args.native_only:
  save('results/READINESS.json',dict(status='passed',allowed_scoring_path='native_uncached_only',synthetic_only=True,
    native_separate_depths=depths,cached_equivalence=False,elapsed_seconds=time.time()-start,
    note='Cached optimization rejected by fixed BF16 parity tolerance; formal scores use original whole-sequence no-cache path.'))
  print('NATIVE_UNCACHED_READINESS_PASSED',flush=True);return
 common=pairs[0][0][:-2];shared=scorer.build_shared_prefix(common)
 cache=scorer.compare_cached_uncached(pairs,shared=shared,char_lengths=[4,3,7],score_atol=.01,token_atol=.05)
 assert cache['passed'],cache
 save('results/READINESS.json',dict(status='passed',synthetic_only=True,native_separate_depths=depths,cache_gate=cache,
    elapsed_seconds=time.time()-start,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated()))
 print(json.dumps(dict(status='passed',elapsed_seconds=time.time()-start)),flush=True)
if __name__=='__main__':main()
