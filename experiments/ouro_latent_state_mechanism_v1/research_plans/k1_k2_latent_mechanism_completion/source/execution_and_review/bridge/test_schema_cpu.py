"""CPU JSON/schema binding faults; no GPU/model or imported producer forwards."""
import json,tempfile,pathlib,importlib.util
p=pathlib.Path(__file__).with_name('b2_freeze_science.py');sp=importlib.util.spec_from_file_location('freeze',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
trace={9:{'input_nonR_sha256':'a','output_nonR_sha256':'b'}}
saved=json.loads(json.dumps({'N_trace':{str(k):v for k,v in trace.items()}}))
assert {str(k):v for k,v in trace.items()}=={str(k):v for k,v in saved['N_trace'].items()}
with tempfile.TemporaryDirectory() as td:
 r=pathlib.Path(td);(r/'pending.json').write_text(json.dumps({'status':'COMPLETE'}));(r/'audit.json').write_text(json.dumps({'status':'PASS','pending_sha256':m.sha(r/'pending.json')}));(r/'analysis.json').write_text(json.dumps({'status':'COMPLETE','audit_sha256':m.sha(r/'audit.json')}))
 receipts={k:{'path':k+'.json','sha256':m.sha(r/(k+'.json')),'status':status,'bindings':binding} for k,status,binding in [('pending','COMPLETE',{}),('audit','PASS',{'pending_sha256':'pending.json'}),('analysis','COMPLETE',{'audit_sha256':'audit.json'})]}
 m.verify(r,receipts)
 # Even updating outer receipt hash cannot authorize a false dependency hash.
 v=json.loads((r/'audit.json').read_text());v['pending_sha256']='0'*64;(r/'audit.json').write_text(json.dumps(v));receipts['audit']['sha256']=m.sha(r/'audit.json')
 try:m.verify(r,receipts);raise RuntimeError('false binding accepted')
 except AssertionError:pass
 # Unknown status cannot pass the exact expected status check.
 v['status']='PASS_BUT_NOT_VERIFIED';(r/'audit.json').write_text(json.dumps(v));receipts['audit']['sha256']=m.sha(r/'audit.json')
 try:m.verify(r,receipts);raise RuntimeError('false status accepted')
 except AssertionError:pass
print('PASS: JSON trace roundtrip, pending/audit/analysis hash chain, forged dependency and status rejected')
