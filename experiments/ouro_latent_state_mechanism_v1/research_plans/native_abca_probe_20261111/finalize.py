"""Publish only after final source and replay guards pass."""
import json
from persistence import write_once,sha
from provenance import verify_replay_ledger

def publish(out,result,runtime,manifest_sha,ledger_sha,ledger,guard):
 out=__import__('pathlib').Path(out)
 payload={'status':'PENDING_INDEPENDENT_REVIEW','runtime':runtime,'manifest_sha256':manifest_sha,'replay_ledger_sha256':ledger_sha,'comparison':result}
 encoded=(json.dumps(payload,sort_keys=True,allow_nan=False)+'\n').encode()
 guard()
 if sha(out/'replay_ledger.json')!=ledger_sha:raise ValueError('ledger drift')
 verify_replay_ledger(out,ledger,'cuda')
 guard()
 write_once(out/'comparison.json',encoded)
