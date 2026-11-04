"""R24 synthetic evidence/review draft only; no real support/tokenizer load."""
import ast,json,sys,platform
from r24_common import ROOT,read,sha
from r24_cpu_checks import main
if __name__=='__main__':
 fs=sorted((ROOT/'scripts').glob('r24_*.py'))
 for p in fs:ast.parse(p.read_text(encoding='utf-8-sig'))
 result=main();result.update(source_sha256={p.name:sha(str(p.relative_to(ROOT))) for p in fs},versions=dict(python=sys.version,platform=platform.platform(),executable=sys.executable))
 (ROOT/'results/r24_CPU_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
 paths=['r24_ASTRA_MEMORY_SUPPORT_DESIGN.md','r24_MEMORY_SCOPE_CLARIFICATION.md','configs/r24_INPUT_MANIFEST.json','results/r24_CPU_CHECKS.json','data/slcd_fixed_demonstrations.json','results/slcd_SOURCE_AUDIT.json','scripts/runtime.py','scripts/r20_spans.py','scripts/r7_references.py']
 paths+=['data/slcd_'+s+'.jsonl' for s in ['discovery','selection']]
 paths+=['results/r7_'+s+'_development'+suffix for s in ['discovery','selection'] for suffix in ['.jsonl','_COMPLETE.json']]
 hashes={p:sha(p) for p in paths};identity=read('configs/r20_VALUE_ABLATION_PILOT_LOCK.json')['model_identity'];hashes.update({'models/Ouro-1.4B/'+p:h for p,h in identity['metadata_and_tokenizer_sha256'].items() if p in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.json','merges.txt','config.json']})
 spec=dict(status='DRAFT_PENDING_ROOT_ASTRA_REVIEW',source_sha256={p.name:sha(str(p.relative_to(ROOT))) for p in fs},input_sha256=hashes,scope='PhaseA CPU support diagnostic only; no real support computed yet, model/states/keys/query/GPU/newdatasets forbidden',cohort='All576 discovery384selection192, alloriginalcandidates, nofilter',values='25knownpastdemoanswers only; answerfield includesoneleading separator space afterAnswer: excludes newline; offsets completelycontained; sourcepositioni>0 keyprospecti-1 recorded only',specific='demoDF<=2 and decodedtrim contains>=2Unicodealnum; diagnostic only, no futurevaluefilter',support='set intersection Sall/Sspecific and candidate-minusR6 targetset difference; actualtargetIDs used offline only',admission='R6wrong goldspecificdifference>=20% combined>=10% eachsplit;>=10specifictypes from>=5demos; allaudits; no pvalues or semantic/fullVreachabilityclaims',separation='supportworker sanitized nogold inputs; admission readslabels only after complete576 clean audited receipt',CPU_wall_limit_seconds=600,GPU_worker_seconds=0,existing_GPU_total=18641.79952767305,existing_GPU_remaining=2958.2004723269492,tokenizer='pinned local_files_only fasttokenizer, originalencode_pair exactprefix/targetdigest, no modelweights',future_scope='prospective native4 fullV Hellinger nearestneighbor per clarification only; no futurekey/query ormemorypilot authorization')
 (ROOT/'configs/r24_REVIEW_DRAFT.json').write_text(json.dumps(spec,indent=2)+'\n');print('DRAFT',sha('configs/r24_REVIEW_DRAFT.json'));print('CPU',sha('results/r24_CPU_CHECKS.json'))
