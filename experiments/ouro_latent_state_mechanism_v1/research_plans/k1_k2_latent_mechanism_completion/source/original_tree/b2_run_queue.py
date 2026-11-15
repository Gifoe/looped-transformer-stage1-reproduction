import pathlib,json,hashlib,subprocess,os,time,traceback
R=pathlib.Path(__file__).resolve().parent;PY='/root/autodl-tmp/ouro-e6a/venv/bin/python'
ENV=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONPATH=str(R/'code')+':'+str(R/'deps')+':'+str(R/'bridge'))
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def state(status,**kw):
 p=R/'B2_QUEUE_STATE.json';q=p.with_suffix('.partial');q.write_text(json.dumps(dict(status=status,epoch=time.time(),**kw),indent=2));os.replace(q,p)
def step(name,args):
 state('RUNNING',step=name)
 with (R/('b2_queue_'+name+'.log')).open('x') as log:subprocess.run([PY,*map(str,args)],env=ENV,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
def manifest(kind,stage):
 f=json.loads((R/'B2_FULL_SOURCE_FREEZE.json').read_text());d=R/'b2_data'/kind/stage;p=json.loads((d/'PENDING.json').read_text());ids=f['reference_ids'][stage] if kind=='references' else f['cohorts'][stage]
 assert p['status']=='COMPLETE' and p['n']==len(ids) and [c['id'] for c in p['cases']]==ids and len(set(ids))==len(ids)
 assert p['source_freeze_sha256']==sha(R/'B2_FULL_SOURCE_FREEZE.json')
 (d/'MANIFEST_COVERAGE.json').write_text(json.dumps(dict(status='PASS',n=len(ids),pending_sha256=sha(d/'PENDING.json'),expected_ID_list_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()),indent=2))
def phase(kind,stage):
 collectorstage='train_all' if kind=='references' and stage=='train' else 'validation299' if kind=='references' else stage
 step(kind+'_'+stage,[R/'bridge/b2_collect.py','--root',R,'--kind','reference' if kind=='references' else 'treatment','--stage',collectorstage])
 manifest(kind,stage)
 step('audit_'+kind+'_'+stage,[R/'bridge/b2_audit.py','--root',R,'--kind',kind,'--stage',stage])
 if kind=='treatments':step('analyze_'+stage,[R/'bridge/b2_analyze.py','--root',R,'--stage',stage])
def main():
 assert not (R/'B2_QUEUE_STATE.json').exists()
 phase('references','train');phase('treatments','train64');phase('treatments','train1030');phase('references','validation');phase('treatments','validation299')
 state('COMPLETE',scope='Final25 collection and declared saved-evidence audits/analyses complete. Scientific synthesis and publication remain.')
if __name__=='__main__':
 try:main()
 except BaseException as ex:state('FAILED_STOPPED',error=repr(ex),traceback=traceback.format_exc());raise
