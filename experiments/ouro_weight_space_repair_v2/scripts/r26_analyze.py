"""Independent saved-vector audit, no labels or new forwards."""
from r26_common import ROOT,gate,read,sha,save,LOCK
from r26_protocol import interpret

def audit_numeric(m,arrays,domain):
 import numpy as np
 a,b=arrays
 unequal=int(np.count_nonzero(a!=b)) if domain=='state' else int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))
 av=(a.astype(np.uint32)<<16).view(np.float32) if domain=='state' else a
 bv=(b.astype(np.uint32)<<16).view(np.float32) if domain=='state' else b
 assert np.isfinite(av).all() and np.isfinite(bv).all()
 d=av.astype(np.float64)-bv.astype(np.float64);maximum=float(np.abs(d).max());rms=float(np.sqrt(np.mean(d*d,dtype=np.float64)))
 assert m['exact']==(unequal==0) and m['unequal_elements']==unequal and m['max_abs']==maximum
 # RMS is independently reported, not required bit-identical to GPU reduction.
 assert np.isfinite(m['RMS']) and 0<=m['RMS']<=maximum
 return rms

def main():
 spec=gate();c=read('results/r26_COMPLETE.json');s=read('logs/r26_SUPERVISOR.json')
 assert c['status']=='complete' and c['contexts']==5 and c['forwards']==30 and c['protocol_sha256']==sha(LOCK)
 assert s['status']=='complete' and s['exit_code']==0 and not s['overrun'] and not s.get('timed_out',False)
 actual=s['gpu_worker_seconds'];assert 0<actual<=120
 assert isinstance(c['all_parameter_hashes_before'],dict) and c['all_parameter_hashes_before'] and c['all_parameter_hashes_before']==c['all_parameter_hashes_after']
 ledgers=spec['budget']['inherited_ledger_sha256'];assert len(ledgers)==14
 for p,h in ledgers.items():assert sha(p)==h
 inherited=sum(read(p)['used_seconds'] for p in ledgers);assert inherited==spec['budget']['inherited_total_seconds']==18819.051603254396
 book=read('logs/r26_gpu_budget.json');assert book['used_seconds']==actual and book['inherited_total_seconds']==inherited and book['total_used_seconds']==inherited+actual<=21600
 assert c['vectors_sha256']==sha('results/r26_VECTORS.npz')
 assert c['raw_sha256']==sha('results/r26_RAW.json') and c['input_audit_sha256']==sha('results/r26_INPUT_AUDIT.json')
 records=read('results/r26_RAW.json')['records'];audit=read('results/r26_INPUT_AUDIT.json');assert audit==read('results/r26_CPU_INPUT_PREFLIGHT.json')
 assert len(records)==len(audit['plan']['contexts'])==5
 for r,p in zip(records,audit['plan']['contexts']):assert r['context']==p['name'] and r['depth']==p['depth'] and r['cuts']==p['cuts'] and r['inputs'][0]['input_ids']==p['tokens']
 import numpy as np,hashlib
 numeric=[]
 with np.load(ROOT/'results/r26_VECTORS.npz',allow_pickle=False) as z:
  referenced=set()
  for r in records:
   for pair in r['comparisons']:
    for domain in ['state','head']:
     m=pair[domain];arrays=[]
     for side in ['A','B']:
      key=m['vector_'+side];a=z[key];referenced.add(key)
      assert list(a.shape)==m['shape']==[1,1,2048 if domain=='state' else 49152]
      assert a.dtype==np.dtype('uint16' if domain=='state' else 'float32')
      assert hashlib.sha256((m['dtype']+str(tuple(m['shape']))).encode()+a.tobytes()).hexdigest()==key
      assert hashlib.sha256(a.tobytes()).hexdigest()==m['sha256_'+side]
      arrays.append(a)
     rms=audit_numeric(m,arrays,domain)
     numeric.append(dict(context=r['context'],variant=pair['variant'],depth=pair['depth'],position=pair['position'],domain=domain,independent_CPU_FP64_RMS=rms,worker_GPU_FP64_RMS=m['RMS'],RMS_abs_difference=abs(rms-m['RMS'])))
  assert referenced==set(z.files) and len(referenced)==c['unique_vectors']
 out=interpret(records);out.update(protocol_sha256=sha(LOCK),raw_sha256=c['raw_sha256'],vectors_sha256=c['vectors_sha256'],independent_vector_numeric_audit=numeric,RMS_scope='CPU NumPy float64 recomputation reported independently; no equality threshold or result-dependent tolerance; exact/unequal/max must match exactly',actual_seconds=actual)
 save('results/r26_ACCEPTANCE.json',out)
if __name__=='__main__':main()
