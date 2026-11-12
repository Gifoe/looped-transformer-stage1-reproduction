import tempfile,json
from pathlib import Path
from unittest.mock import patch
from unit_history import bootstrap,record_active
clean={'LoadState':'not-found','ActiveState':'inactive','ControlGroup':'','Job':''}
with tempfile.TemporaryDirectory() as td:
 root=Path(td);state=root/'state';cg=root/'cg';cg.mkdir()
 with patch('unit_history.fields',return_value=clean):
  bootstrap('ouro-abca-test.service',state,cg)
  try:bootstrap('ouro-abca-test.service',state,cg)
  except RuntimeError:pass
  else:raise AssertionError('pending crash retry accepted')
with tempfile.TemporaryDirectory() as td:
 root=Path(td);cg=root/'cg';cg.mkdir();(cg/'ouro-abca-test.service').mkdir()
 with patch('unit_history.fields',return_value=clean):
  try:bootstrap('ouro-abca-test.service',root/'state',cg)
  except RuntimeError:pass
  else:raise AssertionError('old cgroup accepted')
with tempfile.TemporaryDirectory() as td:
 root=Path(td);cg=root/'cg';cg.mkdir();state=root/'state';unit='ouro-abca-test.service'
 with patch('unit_history.fields',return_value=clean):bootstrap(unit,state,cg)
 expected='/user/app.slice/'+unit;(cg/expected.lstrip('/')).mkdir(parents=True)
 active={'ActiveState':'active','ControlGroup':expected,'Restart':'no'}
 with patch('unit_history.fields',return_value=active):
  record_active(unit,state,cg)
  assert json.loads((state/'previous_unit_controlgroup.json').read_text())['control_group']==expected
  try:record_active(unit,state,cg)
  except RuntimeError:pass
  else:raise AssertionError('record overwrite')
print('BOOTSTRAP_PENDING_CPU_PASS; NOT SYSTEMD OR GPU')
