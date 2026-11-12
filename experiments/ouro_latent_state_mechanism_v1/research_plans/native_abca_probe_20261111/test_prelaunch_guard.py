import tempfile
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
from prelaunch_guard import prior_absent
with tempfile.TemporaryDirectory() as td:
 root=Path(td);unit='ouro-abca-fixture.service';cg='/user/app.slice/'+unit;p=root/cg.lstrip('/');p.mkdir(parents=True);events=p/'cgroup.events'
 with patch('prelaunch_guard.subprocess.run',return_value=SimpleNamespace(returncode=0,stdout='ActiveState=failed\nControlGroup=\nRestart=no\nJob=\n')):
  assert prior_absent(unit,cg,root) is False
  events.write_text('malformed\n');assert prior_absent(unit,cg,root) is False
  events.write_text('populated 1\n');assert prior_absent(unit,cg,root) is False
  events.write_text('populated 0\n');assert prior_absent(unit,cg,root) is True
print('POPULATED_PRIOR_CGROUP_MOCK_PASS; NOT SYSTEMD RACE')
