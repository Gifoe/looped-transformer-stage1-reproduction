"""Post-real-technical-gate CPU proof. No science-freeze dependency or GPU."""
import argparse,pathlib,sys,json,hashlib,numpy as np
from b2_audit import audit_case,score,sha

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--tag',default='b2_technical');a=ap.parse_args();assert np.finfo(np.longdouble).nmant>52,'Extended-precision independent reduction required';root=pathlib.Path(a.root).resolve();sys.path[:0]=[str(root/'code'),str(root/'deps'),str(root/'storage')];from ab1_prefix_codec import load
 d=root/a.tag;gp=d/'TECHNICAL_GATE.json';g=json.loads(gp.read_text());assert g['status']=='B2_THREE_TECHNICAL_AVAILABLE_ENDPOINT_WIRING_PASS';tf=root/'B2_TECH_SOURCE_FREEZE.json';assert sha(tf)==g['technical_freeze_sha256'];sources=dict(g['source_sha256']);assert all(sha(root/k)==v for k,v in sources.items());assert 'bridge/b2_technical_cpu_audit.py' in sources and 'bridge/b2_audit.py' in sources
 for p in (gp,tf,pathlib.Path(__file__),root/'bridge/b2_audit.py',root/'data/train64/PENDING.json',root/'data/train64/INDEPENDENT_COVERAGE_AUDIT.json'):sources[str(p.relative_to(root))]=sha(p)
 pending=json.loads((root/'data/train64/PENDING.json').read_text());ab_audit=json.loads((root/'data/train64/INDEPENDENT_COVERAGE_AUDIT.json').read_text());assert ab_audit['status']=='PASS' and ab_audit['pending_sha256']==sha(root/'data/train64/PENDING.json');ab_cases={c['id']:c for c in pending['cases']};assert len(g['cases'])==3 and len({c['id'] for c in g['cases']})==3;data={};hashes={}
 for c in g['cases']:
  mp=d/c['metadata_file'];assert sha(mp)==c['metadata_sha256'];m=json.loads(mp.read_text());t=load(m,d);abc=ab_cases[c['id']];abp=root/'data/train64'/abc['metadata_file'];assert sha(abp)==abc['metadata_sha256']==m['AB_metadata_sha256'];abm=json.loads(abp.read_text());ab=load(abm,abp.parent);assert m['id']==abm['id'] and m['split']==abm['split'];sources[str(mp.relative_to(root))]=sha(mp);sources[str(abp.relative_to(root))]=sha(abp);data[c['id']]=(m,t,abm,ab);hashes[c['id']]=sha(mp)
 records=[]
 for ident,(m,t,abm,ab) in data.items():
  dm,dt,dabm,dab=data[m['donor_id']];assert dm['id']!=ident and dm['split']==m['split'];ref=m['reference'];treatment=m['treatments'];assert ref['source_identity']==treatment['source_identity']==abm['source_identity'];targets=abm['targets']
  for key,sc in [('native_K1_logits',abm['identity']['K1_scores']),('native_K2_logits',abm['identity']['N_scores']),('H01_logits',abm['conditions']['H01']['scores'])]:score(ab[key],sc,targets)
  rm=dict(ref,targets=targets);tm=dict(treatment,targets=targets);rr=audit_case(rm,t,ab,ref,t,reference_only=True);tr=audit_case(tm,t,ab,ref,t,dab,dt);records.append(dict(id=ident,reference=rr,treatments=tr,source_endpoints=['K1','N','U'],expected_scientific_endpoints=25))
 assert all(sha(root/k)==v for k,v in sources.items());receipt=dict(status='PASS',longdouble_nmant=int(np.finfo(np.longdouble).nmant),technical_gate_sha256=sha(gp),technical_source_freeze_sha256=sha(tf),case_metadata_sha256=hashes,source_sha256=sources,records=records,scope='Independent CPU archived-score/control mathematics for actual three-case technical data. No independent training, new model forward or whole-goal claim; undefined retained only with independently proven zero predicates.');out=d/'INDEPENDENT_CPU_PROOF.json';assert not out.exists();out.write_text(json.dumps(receipt,indent=2)+'\n');print('PASS')
if __name__=='__main__':main()
