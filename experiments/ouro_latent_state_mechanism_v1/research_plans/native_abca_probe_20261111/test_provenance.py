import json,tempfile
from pathlib import Path
from provenance import freeze,verify,pqr_complete,sha,verify_replay_ledger
with tempfile.TemporaryDirectory() as td:
 d=Path(td);p=d/'code';p.write_text('v1');m=freeze([p],{'model':'hash'},{'torch':'v'});verify(m,[p],{'model':'hash'},{'torch':'v'});p.write_text('v2')
 try:verify(m,[p],{'model':'hash'},{'torch':'v'})
 except ValueError:pass
 else:raise AssertionError('source drift')
 (d/'status.json').write_text(json.dumps({'status':'FAILED'}))
 try:pqr_complete(d,'b',{'train':{'x'}})
 except ValueError:pass
 else:raise AssertionError('failed run accepted')
 (d/'train').mkdir();receipt=d/'train'/'_receipt.json';receipt.write_text(json.dumps({'status':'PQR_SPLIT_PENDING_COVERAGE','split':'train','questions':1,'binding_sha256':'b'}))
 report={'status':'PQR_COMPLETE_CUDA_REPLAY_ZERO','split':'train','expected_questions':1,'complete_questions':1,'replayed_condition_logits':4,'binding_sha256':'b','questions':[{'id':'x','metadata_sha256':'a'*64,'logits_sha256':'c'*64}],'receipt_sha256':sha(receipt)};cv=d/'coverage_train.json';cv.write_text(json.dumps(report))
 analysis=d/'analysis.json';a={'status':'PQR_FULL_COVERAGE_DESCRIPTIVE_PENDING_REVIEW','binding_sha256':'b','coverage_sha256':{'train':sha(cv)},'splits':{'train':{'all':{'n':1}}}};analysis.write_text(json.dumps(a));st={'status':'PQR_COMPLETE_PENDING_ASTRA_REVIEW','analysis_path':str(analysis),'analysis_sha256':sha(analysis)};(d/'status.json').write_text(json.dumps(st));pqr_complete(d,'b',{'train':{'x'}})
 try:pqr_complete(d,'b',{'train':{'y'}})
 except ValueError:pass
 else:raise AssertionError('wrong cohort')
 receipt.write_text('{}');report['receipt_sha256']=sha(receipt);cv.write_text(json.dumps(report));a['coverage_sha256']['train']=sha(cv);analysis.write_text(json.dumps(a));st['analysis_sha256']=sha(analysis);(d/'status.json').write_text(json.dumps(st))
 try:pqr_complete(d,'b',{'train':{'x'}})
 except ValueError:pass
 else:raise AssertionError('empty receipt')
 receipt.write_text(json.dumps({'status':'PQR_SPLIT_PENDING_COVERAGE','split':'train','questions':1,'binding_sha256':'b'}));report['receipt_sha256']=sha(receipt);cv.write_text(json.dumps(report))
 try:pqr_complete(d,'b',{'train':{'x'}})
 except ValueError:pass
 else:raise AssertionError('analysis coverage mixed')
with tempfile.TemporaryDirectory() as td:
 d=Path(td);ledger={}
 for label in ('A1','B','C','A2'):
  (d/(label+'.json')).write_text(label);(d/(label+'.safetensors')).write_text(label)
  ledger[label]={'scores_zero_difference':True,'replay_device':'cuda','receipt_sha256':sha(d/(label+'.json')),'tensor_sha256':sha(d/(label+'.safetensors'))}
 verify_replay_ledger(d,ledger,'cuda');(d/'B.json').write_text('tampered')
 try:verify_replay_ledger(d,ledger,'cuda')
 except ValueError:pass
 else:raise AssertionError('replay evidence drift')
print('PROVENANCE_TOY_PASS; NOT GPU')
