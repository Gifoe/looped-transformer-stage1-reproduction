"""Generate reviewable final reports/figures after all original gates, never publish."""
import hashlib,json,os,pathlib,subprocess,time,traceback
R=pathlib.Path(__file__).resolve().parents[1];D=R/'reporting';PY='/root/autodl-tmp/ouro-e6a/venv/bin/python'
read=lambda p:json.loads(pathlib.Path(p).read_text())
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
scripts=['build_reports.py','plot_results.py','write_readme.py'];sources={s:sha(D/s) for s in scripts}
def state(status,**fields):
 p=D/'QUEUE_STATE.json';tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(dict(status=status,epoch=time.time(),sources=sources,**fields),indent=2));os.replace(tmp,p)
try:
 assert not (D/'QUEUE_STATE.json').exists()
 state('WAITING_FOR_ALL_INDEPENDENT_AUDITS')
 while True:
  a=read(R/'B2_QUEUE_STATE.json');b=read(R/'posthoc_review/QUEUE_STATE.json')
  assert a['status']!='FAILED_STOPPED' and b['status']!='FAILED_STOPPED'
  c=R/'parallel/ALL_LINEAGE_COMPLETE.json'
  if a['status']=='COMPLETE' and b['status']=='COMPLETE' and c.exists() and read(c)['status']=='PASS':break
  time.sleep(15)
 for script in scripts:
  assert all(sha(D/s)==h for s,h in sources.items())
  state('RUNNING',step=script)
  with (D/(script+'.log')).open('x') as log:subprocess.run([PY,str(D/script)],cwd=R,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
 state('REPORTS_AND_FIGURES_GENERATED_AWAITING_VISUAL_QA',scope='Not GitHub published. Package/reproducer and final claim audit still required.')
except BaseException as error:
 state('FAILED_STOPPED',error=repr(error),traceback=traceback.format_exc());raise
