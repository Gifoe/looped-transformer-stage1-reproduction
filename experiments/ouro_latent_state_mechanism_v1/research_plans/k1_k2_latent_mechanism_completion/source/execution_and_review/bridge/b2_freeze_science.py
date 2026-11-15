"""Final25 prospective freeze creator. No GPU operations or launch."""
import argparse,json,pathlib,hashlib,time,sys
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_text())
def verify(root,receipts):
 for name,r in receipts.items():
  p=root/r['path'];assert sha(p)==r['sha256'],name
  v=read(p);assert v['status']==r['status'],name
  for field,dependency in r.get('bindings',{}).items():assert v[field]==sha(root/dependency),(name,field)
def verify_technical(root,r):
 sys.path[:0]=[str(root/'code'),str(root/'deps'),str(root/'bridge')]
 from ab1_prefix_codec import load
 gp=root/r['path'];g=read(gp)
 assert g['status']=='B2_THREE_TECHNICAL_AVAILABLE_ENDPOINT_WIRING_PASS'
 tf=root/r['technical_freeze_path'];assert sha(tf)==g['technical_freeze_sha256']
 for path,digest in g['source_sha256'].items():assert sha(root/path)==digest
 assert len(g['cases'])==3 and len({x['id'] for x in g['cases']})==3
 expected={'N','U','T','TX','TA','TXA','U_plus','upstream_norm','upstream_unrelated','U_plus_norm_preserve','U_plus_unrelated','M','U_M_plus','M_norm_preserve','M_unrelated','U_M_plus_norm_preserve','U_M_plus_unrelated'}|{f'{n}_{b}_{k}' for n in ('T','N') for b in ('X','A') for k in ('norm','unrelated')}
 assert len(expected)==25
 for c in g['cases']:
  mp=gp.parent/(c.get('metadata_file') or (hashlib.sha256(c['id'].encode()).hexdigest()+'.json'));assert sha(mp)==c['metadata_sha256'];m=read(mp);load(m,gp.parent)
  assert m['expected_endpoints']==25 and set(m['independent_replayed'])|set(m['undefined_endpoints'])==expected
  assert len(m['independent_replayed'])==m['actual_replayed_count'] and not set(m['independent_replayed'])&set(m['undefined_endpoints'])
 # Separate independent technical CPU proof receipt binds actual technical gate + every case.
 ar=root/r['undefined_proof_audit_path'];av=read(ar);assert sha(ar)==r['undefined_proof_audit_sha256'] and av['status']=='PASS' and av['technical_gate_sha256']==sha(gp)
 assert av['case_metadata_sha256']=={c['id']:c['metadata_sha256'] for c in g['cases']}
 for path,digest in av['source_sha256'].items():assert sha(root/path)==digest
 return {str(tf.relative_to(root)):sha(tf),str(ar.relative_to(root)):sha(ar)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--readiness',required=True);a=p.parse_args();root=pathlib.Path(a.root).resolve();ready_path=pathlib.Path(a.readiness).resolve();assert ready_path.is_relative_to(root);ready_sha=sha(ready_path);ready=read(ready_path)
 required={'technical'}|{f'{s}_{k}' for s in ('train64','train1030','validation299') for k in ('pending','audit','analysis','E2audit')}
 assert required<=set(ready['receipts']);verify(root,ready['receipts'])
 for stage in ('train64','train1030','validation299'):
  r=ready['receipts'];pending=r[stage+'_pending']['path'];audit=r[stage+'_audit']['path']
  assert read(root/pending)['status']=='COMPLETE'
  assert read(root/audit)['status']=='PASS' and read(root/audit)['pending_sha256']==sha(root/pending)
  an=read(root/r[stage+'_analysis']['path']);assert an['status']=='COMPLETE' and an['audit_sha256']==sha(root/audit)
  assert read(root/r[stage+'_E2audit']['path'])['status']=='PASS'
 techsources=verify_technical(root,ready['receipts']['technical'])
 # All expected producer/auditor sources explicitly enumerated; missing analysis source stops freeze.
 for n in ('bridge/b2_kernel.py','bridge/b2_forward.py','bridge/b2_controls.py','bridge/b2_independent.py','bridge/b2_technical_driver.py','bridge/b2_collect.py','bridge/b2_freeze_science.py','bridge/b2_audit.py','bridge/b2_analyze.py'):
  assert n in ready['sources'] and sha(root/n)==ready['sources'][n],n
 ab=read(root/'FULL_SOURCE_FREEZE.json');coh=ab['cohorts'];train=sorted(coh['train64']+coh['train1030']);val=sorted(coh['validation299']);assert len(train)==len(set(train))==1094 and len(val)==len(set(val))==299 and not set(train)&set(val)
 donors={split:{x:ids[(i+1)%len(ids)] for i,x in enumerate(ids)} for split,ids in [('train',train),('validation',val)]}
 assert ready['final_order_path'] in ready['sources'] and ready['queue_path'] in ready['sources'] and ready['codec_path'] in ready['sources']
 sources=dict(ready['sources']);sources.update(techsources);sources[str(ready_path.relative_to(root))]=ready_sha;sources['FULL_SOURCE_FREEZE.json']=sha(root/'FULL_SOURCE_FREEZE.json')
 for name,digest in ab['sources'].items():assert sha(root/name)==digest;sources[name]=digest
 sources.update({r['path']:r['sha256'] for r in ready['receipts'].values()})
 f={'status':'B2_FINAL25_SCIENCE_FROZEN','sources':sources,'readiness':ready['receipts'],'cohorts':coh,'reference_ids':{'train':train,'validation':val},'donors':donors,'gpu_uuid':ab['gpu_uuid'],'model_path':ready['model_path'],'lock_path':ready['lock_path'],'utc_epoch':time.time(),'contract':{'endpoints':25,'treatments':22,'families':{'attention_upstream':6,'L18branch':12,'MLPsecondary':6},'bootstrap_replicates':5000,'bootstrap_seed':20261114,'correction':'separate Bonferroni per family','primary':'fresh same-site natural-rescue; natural-no-tie reversal check; exact BF16 sensitivity','directions':'all predefined positive; 64 engineering/descriptive only; 1030/299 separate','limits':'TXA identity structural; no semantic origin/unique writer claim'}}
 verify(root,ready['receipts']);assert sha(ready_path)==ready_sha
 for path,digest in sources.items():assert sha(root/path)==digest
 out=root/'B2_FULL_SOURCE_FREEZE.json'
 with out.open('x') as h:json.dump(f,h,indent=2,sort_keys=True);h.write('\n')
 print(sha(out))
if __name__=='__main__':main()
