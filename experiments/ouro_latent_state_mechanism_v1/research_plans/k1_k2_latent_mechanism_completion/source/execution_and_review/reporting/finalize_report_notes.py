"""Final scope/provenance clarifications; no numeric or confirmatory-criterion changes."""
import hashlib,json,pathlib
R=pathlib.Path(__file__).resolve().parents[1];O=R/'publication/experiments/ouro_latent_state_mechanism_v1/research_plans/k1_k2_latent_mechanism_completion'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
assert (O/'results/B2/INTERACTION_DESCRIPTIVE.json').exists()
assert not (O/'receipts/REPORT_CLARIFICATIONS.json').exists()
changes=[]
def edit(relative,operation):
 p=O/relative;before=sha(p);s=p.read_text();updated=operation(s);p.write_text(updated)
 changes.append(dict(file=relative,before_sha256=before,after_sha256=sha(p)))
edit('INFORMATION_AVAILABILITY.md',lambda s:s.replace('unsaved L2-L8 prevent a first-physical-writer claim','unobserved layer outputs between the L1 input and L9 input prevent a first-physical-writer claim'))
note='\nThe endpoint name `unrelated` denotes the prospectively fixed next-lexicographic different-question direction. This is a tested other-question directional control, not a certificate of semantic independence. No labels or outcomes choose/resample the donor.\n'
edit('UPSTREAM_CAUSAL_TRACING.md',lambda s:s+note)
edit('NATURAL_MEDIATION_RESULTS.md',lambda s:s+note)
num=json.loads(pathlib.Path('/root/autodl-tmp/ouro-k1-k2-mechanism/evidence_packet/NUMERICAL8_INDEPENDENT_COMPARISON.json').read_text());groups={g:sum(c['native_BF16_group']==g for c in num['cases']) for g in set(c['native_BF16_group'] for c in num['cases'])};changed=sum(c['endpoints']['N']['prediction_changed'] for c in num['cases'])
edit('NUMERICAL_AND_SOURCE_AUDIT.md',lambda s:s+'\nThe fixed8 historical numerical subset contains '+json.dumps(groups,sort_keys=True)+'; natural K2 predictions changed in '+str(changed)+'/8 cases. Other natural groups are not numerically sampled there, and this is not a new B2 robustness experiment.\n')
edit('README.md',lambda s:s.replace('original B2 contrasts','original AB1 primary intervals/gates, B2 contrasts').replace('original5000-bootstrap intervals/gates','original5000-bootstrap intervals/gates and the separately labeled descriptive factorial interaction'))
# Exact case-level raw provenance, joined without selection by original stage/ID.
maps={s:{c['id']:c for c in json.loads((R/'b2_data/treatments'/s/'PENDING.json').read_text())['cases']} for s in ('train64','train1030','validation299')}
p=O/'results/COUNTEREXAMPLE_EXAMPLES.jsonl';before=sha(p);records=[]
for line in p.read_text().splitlines():
 r=json.loads(line);r['case_provenance']=maps[r['stage']][r['id']];r['source_analysis_rows_sha256']=sha(R/'b2_analysis'/r['stage']/'rows.jsonl');records.append(r)
p.write_text(''.join(json.dumps(r)+'\n' for r in records));changes.append(dict(file=str(p.relative_to(O)),before_sha256=before,after_sha256=sha(p)))
(O/'receipts').mkdir(exist_ok=True)
(O/'receipts/REPORT_CLARIFICATIONS.json').write_text(json.dumps(dict(status='COMPLETE',script_sha256=sha(__file__),changes=changes,scope='Correct saved-site interpretation, distinguish different-question from semantic independence, state numerical subset and join exact counterexample provenance. No intervention/statistical-criterion change.'),indent=2))
print('REPORT_SCOPE_AND_CASE_PROVENANCE_FINALIZED')
