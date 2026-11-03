"""Bounded local orchestration of our own two diagnostic stages, detached."""
import json,subprocess,time
from runtime import ROOT,save

def main():
 deadline=time.time()+3000
 receipt=ROOT/'logs/layer_screen_SUPERVISOR.json'
 while True:
  state=json.loads(receipt.read_text())
  if state['status'] in ['complete','failed']:break
  if time.time()>deadline:raise RuntimeError('Own diagnostic supervisor did not finish in its bound')
  time.sleep(5)
 assert state['status']=='complete',state
 with (ROOT/'logs/layer_analysis.out').open('x') as out,(ROOT/'logs/layer_analysis.err').open('x') as err:
  subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'scripts/analyze_layer_screen.py')],cwd=ROOT,stdout=out,stderr=err,check=True,timeout=120)
 with (ROOT/'logs/shared_confirmation_launcher.out').open('x') as out,(ROOT/'logs/shared_confirmation_launcher.err').open('x') as err:
  subprocess.run([str(ROOT/'env/bin/python'),str(ROOT/'scripts/supervise.py'),'--name','shared_confirmation','--seconds','600','--','run_shared_confirmation.py'],cwd=ROOT,stdout=out,stderr=err,check=True,timeout=650)
 save('logs/AFTER_LAYER_PIPELINE.json',dict(status='complete',finished_unix=time.time()))
if __name__=='__main__':main()
