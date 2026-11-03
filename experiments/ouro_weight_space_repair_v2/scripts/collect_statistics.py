"""Forward-only all-site statistics from training questions; no activation lesions."""
import hashlib,json,time
from runtime import ROOT,load_model,encode_pair,outcome,save,check_budget
from ouro_cached_scoring import OuroCachedScorer

def filehash(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class Statistics:
 def __init__(self,model):
  import torch
  self.model=model;self.handles=[];self.indices=None;self.loop=0;self.seen=0;self.weight=1.;self.buffers={}
  def layer0(module,args,kwargs):self.loop=int(kwargs['current_ut'])+1
  self.handles.append(model.model.layers[0].register_forward_pre_hook(layer0,with_kwargs=True))
  for l,layer in enumerate(model.model.layers):
   for branch,projection in [('mlp',layer.mlp.down_proj),('attention',layer.self_attn.o_proj)]:
    assert projection.weight.shape[0]==2048
    def hook(module,args,output,l=l,branch=branch):
     if self.loop not in [3,4,5,6]:return
     x=args[0].detach().index_select(1,self.indices).float();y=output.detach().index_select(1,self.indices).float()
     self.buffers[branch+'_input_second_moment'][l,self.loop-3].add_(x.square().mean((0,1)),alpha=self.weight)
     self.buffers[branch+'_raw_output_mean'][l,self.loop-3].add_(y.mean((0,1)),alpha=self.weight)
     self.seen+=1
    self.handles.append(projection.register_forward_hook(hook))
 def reset_question(self,nchoices):
  import torch
  self.weight=1/nchoices;self.buffers={}
  for branch,width in [('mlp',self.model.model.layers[0].mlp.down_proj.weight.shape[1]),('attention',self.model.model.layers[0].self_attn.o_proj.weight.shape[1])]:
   self.buffers[branch+'_input_second_moment']=torch.zeros((24,4,width),device='cuda',dtype=torch.float32)
   self.buffers[branch+'_raw_output_mean']=torch.zeros((24,4,2048),device='cuda',dtype=torch.float32)
 def close(self):
  for h in self.handles:h.remove()

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--n',type=int,default=384);a=ap.parse_args();assert a.n in [2,384]
 rows=[json.loads(l) for l in (ROOT/'data/discovery_expanded.jsonl').read_text().splitlines()][:a.n]
 design=json.loads((ROOT/'configs/CANDIDATE_DESIGN.json').read_text());assert design['candidate_count']==14
 codehash=hashlib.sha256(''.join(filehash(ROOT/'scripts'/p) for p in ['runtime.py','ouro_cached_scoring.py','collect_statistics.py']).encode()).hexdigest()
 out=ROOT/'results/training_statistics';out.mkdir(exist_ok=True)
 source={r['id']:r for r in map(json.loads,(ROOT/'data/v1_reference_discovery_scores.jsonl').read_text().splitlines())}
 path=out/'scores.jsonl';done={}
 if path.exists():
  for l in path.read_text().splitlines():
   r=json.loads(l);assert r['implementation_sha256']==codehash;assert r['id'] not in done;done[r['id']]=r
 start=time.time();tok,model=load_model();scorer=OuroCachedScorer(model,6)
 import torch
 collector=Statistics(model);scorer.invalidate();parity=[]
 try:
  for row in rows:
   if row['id'] in done:continue
   check_budget(90);pairs=[encode_pair(tok,*p) for p in row['request_pairs']];collector.reset_question(len(pairs));values=[];states=[];beg=time.time()
   for j,(prefix,target) in enumerate(pairs):
    collector.indices=torch.arange(len(prefix)-1,len(prefix)+len(target)-1,device='cuda',dtype=torch.long);collector.seen=0
    value=scorer.score_uncached([(prefix,target)],capture_raw=True,return_states=True)[0]
    assert collector.seen==24*2*4
    values.append(value);states.append(dict(candidate=j,target_ids=target,normalized=value['normalized_states'],raw=value['raw_states']))
   methods={f'native_k{k}':outcome([v['sum_loglikelihood'][k] for v in values],row['choice_char_lengths'],row['gold']) for k in range(1,7)}
   for name in ['fixed','adaptive']:methods['loopcd_k4_'+name]=outcome([v['loopcd'][name] for v in values],row['choice_char_lengths'],row['gold'])
   if row['id'] in source and len(parity)<2:
    errors={name:max(abs(x-y) for x,y in zip(v['sum_scores'],source[row['id']]['methods'][name]['sum_scores'])) for name,v in methods.items()}
    assert all(v==0 for v in errors.values()),('V1 original/recording parity failed',row['id'],errors)
    parity.append(dict(id=row['id'],errors=errors))
   stem=row['id'].rsplit('/',1)[-1];payload={k:v.cpu() for k,v in collector.buffers.items()}
   torch.save(dict(id=row['id'],loops=[3,4,5,6],statistics=payload,candidate_states=states,averaging='target-token means, equal candidate weights',implementation_sha256=codehash),out/(stem+'.pt'))
   r=dict(id=row['id'],gold=row['gold'],source_index=row['source_index'],scope='train_discovery_only',prompt_sha256=row['prompt_sha256'],choice_char_lengths=row['choice_char_lengths'],choice_token_counts=[len(t) for _,t in pairs],methods=methods,implementation_sha256=codehash,elapsed_seconds=time.time()-beg,statistics_file=stem+'.pt')
   with path.open('a',encoding='utf8') as f:f.write(json.dumps(r)+'\n')
   done[row['id']]=r
   if len(done)%8==0:
    progress=dict(completed=len(done),target=a.n,elapsed_seconds=time.time()-start);save('results/training_statistics/PROGRESS.json',progress);print(json.dumps(progress),flush=True)
 finally:collector.close();scorer.invalidate()
 assert len(done)==a.n
 if not parity and (out/'PARITY.json').exists():parity=json.loads((out/'PARITY.json').read_text())['cases']
 if parity:save('results/training_statistics/PARITY.json',dict(status='passed',exact_recording_and_v1_original=True,cases=parity))
 save(f'results/training_statistics/COMPLETE_{a.n}.json',dict(status='complete',n=a.n,implementation_sha256=codehash,elapsed_seconds=time.time()-start))
 print('TRAINING_STATISTICS_COMPLETE',a.n,flush=True)
if __name__=='__main__':main()
