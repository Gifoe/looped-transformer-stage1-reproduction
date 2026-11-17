import pathlib,json,hashlib,ast,datetime
R=pathlib.Path('/root/ouro-region-use-sol-20261117');D=R.parent/'ouro-binding-interface-sol-20261117';P=R.parent/'ouro-semantic-read-sol-20261117'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (R/'REGION_FREEZE.json').exists();assert not list((R/'results').glob('*.json'));assert json.loads((R/'INPUT_CPU_AUDIT.json').read_text())['status']=='PASS'
for p in (R/'code').glob('*.py'):ast.parse(p.read_text())
files={str(p.relative_to(R)):sha(p) for p in sorted((R/'code').glob('*.py'))};files.update({n:sha(R/n) for n in ['INPUTS.jsonl','FRESH_GROUPS.json','INPUT_CPU_AUDIT.json','PROTOCOL.md','ASTRA_REGION_PROTOCOL_FINAL_REVIEW.md']});inherited={}
for base,freeze in [(P,'SCIENCE_FREEZE.json'),(D,'DISCOVERY_FREEZE.json')]:
 for n,d in json.loads((base/freeze).read_text()).get('files_sha256',json.loads((base/freeze).read_text()).get('source_and_inputs_sha256')).items():assert sha(base/n)==d;inherited[str(base/n)]=d
for p in sorted((D/'results').glob('SR*_L11.json')):inherited[str(p)]=sha(p)
for p in sorted((P/'native_results').glob('SR*.json')):
 if int(p.stem[2:])<16:inherited[str(p)]=sha(p)
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Fixed K2/L11 wholeG idealpairedmessage receiver-region use, prospective fresh confirmation','files_sha256':files,'inherited_files_sha256':inherited,'parent_freeze_sha256':sha(P/'SCIENCE_FREEZE.json'),'discovery_freeze_sha256':sha(D/'DISCOVERY_FREEZE.json'),'published_discovery_commit':'12621805b8ffe9d274f01ff8fa8d7fd5f413d0e8','new_region_model_outcomes_before_freeze':0,'technical_body_forwards':36,'development_new_body_forwards':6144,'confirmation_forwards':{'selected7':6144,'selected_other':10240},'selection':'minimalnumberregions then maxminimum4retentionratio then smallestmask, eachE>.1 and>=.8E7,8positive,doseQC; no fallback afterconfirmation','confirmation_statistics':'18family/100000basegroupbootstraps/seed2026111745/Bonferroni simultaneous95%; no L11 or mask reselection','fresh_seed':2026111743,'scientific_conclusion_scope':'Region use of specified idealmessage intervention, not unique algorithm or native necessity','goal_complete':False};(R/'REGION_FREEZE.json').write_text(json.dumps(out,indent=2));print('REGION_FROZEN',sha(R/'REGION_FREEZE.json'),flush=True)
