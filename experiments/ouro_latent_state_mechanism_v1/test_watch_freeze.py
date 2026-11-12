"""CPU regression for tracked, untracked and non-Git frozen input drift."""
import subprocess,tempfile
from pathlib import Path
import ouro_watch as w
def rejects(fn):
 try:fn()
 except RuntimeError:return
 raise AssertionError('drift accepted')
def main():
 old=(w.ROOT,w.REPO,w.R);python=w.PY
 with tempfile.TemporaryDirectory(prefix='watch_freeze_',dir=old[0]) as d:
  w.ROOT=Path(d);w.REPO=w.ROOT/'repo';w.REPO.mkdir();w.R=w.REPO/'src';w.R.mkdir();source=w.R/'fixture.py';source.write_text('fixture=1\n');data=w.ROOT/'data';data.write_text('first')
  freeze=subprocess.check_output([str(python),'-m','pip','freeze'],text=True);(w.R/'environment_pip_freeze.txt').write_text(freeze)
  subprocess.run(['git','init','-q'],cwd=w.REPO,check=True);subprocess.run(['git','add','src'],cwd=w.REPO,check=True);subprocess.run(['git','-c','user.name=fixture','-c','user.email=fixture@localhost','commit','-qm','fixture'],cwd=w.REPO,check=True);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w.REPO,text=True).strip()
  pre={'source_hashes':{'fixture.py':w.sha(source)},'bound_hashes':{'train':{'data':w.sha(data)}}};w.verify_snapshot(pre,head)
  data.write_text('altered');rejects(lambda:w.verify_snapshot(pre,head));data.write_text('first')
  source.write_text('fixture=2\n');rejects(lambda:w.verify_snapshot(pre,head));source.write_text('fixture=1\n')
  (w.R/'unexpected.py').write_text('x=1\n');rejects(lambda:w.verify_snapshot(pre,head))
 w.ROOT,w.REPO,w.R=old
 print('PASS frozen data, tracked code and untracked code drift rejection')
if __name__=='__main__':main()
