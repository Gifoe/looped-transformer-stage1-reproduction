import pathlib,subprocess,sys,os,json,time,datetime
R=pathlib.Path('/root/ouro-region-use-sol-20261117');PY=sys.executable;env={**os.environ,'CUDA_VISIBLE_DEVICES':'GPU-d4e36609-350a-a3be-5024-3eefc3ad01fc','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'};phase=sys.argv[1];assert phase in ['development','confirmation']
def run(name,args):
 with (R/'logs'/(name+'.log')).open('w') as log:
  p=subprocess.Popen([PY,str(R/'code'/args[0])]+args[1:],env=env,stdout=log,stderr=subprocess.STDOUT);print('RUN',name,p.pid,flush=True);assert p.wait()==0,('FAILED',name)
if phase=='development':
 run('technical',['technical.py']);run('oracle',['oracle.py'])
ps=[];handles=[]
for rank in range(3):
 log=(R/'logs'/(phase+'_rank'+str(rank)+'.log')).open('w');handles.append(log);p=subprocess.Popen([PY,str(R/'code'/('run_development.py' if phase=='development' else 'run_confirmation.py')),str(rank)],env=env,stdout=log,stderr=subprocess.STDOUT);ps.append(p);print('WORKER',phase,rank,p.pid,flush=True)
codes=[p.wait() for p in ps];[f.close() for f in handles];assert codes==[0,0,0],codes
run(phase+'_audit',['audit_vectors.py',phase]);run(phase+'_analysis',['analyze.py',phase]);print('PHASE_COMPLETE',phase,datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
