import pathlib,json,gzip,shutil
R=pathlib.Path('/root/ouro-causal-edge-sol-20261117');P=R/'publication';P.mkdir(exist_ok=True);assert json.loads((R/'CPU_SCALAR_AUDIT.json').read_text())['status']=='PASS'
for folder in ['code','prior']:
 for f in (R/folder).rglob('*'):
  if f.is_file() and f.suffix in ['.py','.md','.json'] and '__pycache__' not in f.parts:
   dest=P/f.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
for f in R.glob('*.md'):
 if f.name!='ACTIVE_RUN.md':shutil.copy2(f,P/f.name)
for f in R.glob('*.json'):
 if f.name=='GITHUB_PUBLICATION_RECEIPT.json':continue
 if f.stat().st_size<=2*1024*1024:shutil.copy2(f,P/f.name)
 else:
  with f.open('rb') as src,gzip.GzipFile(filename=str(P/(f.name+'.gz')),mode='wb',mtime=0) as dst:shutil.copyfileobj(src,dst)
for f in R.glob('*.jsonl'):
 with f.open('rb') as src,gzip.GzipFile(filename=str(P/(f.name+'.gz')),mode='wb',mtime=0) as dst:shutil.copyfileobj(src,dst)
shutil.copy2(R/'code/ATTENTION_EDGE_IMPLEMENTATION.py',P/'ATTENTION_EDGE_IMPLEMENTATION.py')
(P/'README.md').write_text('''# Fixed attention edge discovery

Three K2 paths (S->M L6/L10; S->R L14), three actual post-RoPE Value/Key/KV transfers, native/same-variable/unrelated/equal-realized-dose controls. Complete exposed32developmentbasepairs/128units and fresh64basepairs/256units. Read FINAL_MECHANISTIC_CONCLUSION.md and the twelve requested reports. All work by Sol; no training, Test, head/layer search or Astra agent.

Inputs and symbolic labels were frozen before new outcomes. Every original basepair includes both directions and reversed candidate ordering; inference uses original label tokens. Unrelated controls have a third-box first-hop source absent from the receiver's mappings and are not scored as complete tasks. Source-before-mapping invariance is tested at both passes/allthree layers under mapping and option edits.

CPU replay: Python3.10 + NumPy1.26.4, `python code/reproduce_release.py --release .`. Lossless archive shards are assembled automatically. Recomputes all outcome categories, strict/novel crosses, basepair-cluster confidence intervals, gates and development selection exactly; independently audits input hashes, symbolic crossing, candidates/ties and actual perturbation metadata. This does not rerun neural forwards. Full49152vectors and actualQ/K/V witnesses remain on server under RAW_TENSOR_MANIFEST.json, independently audited by CPU_FULL_VECTOR_AUDIT and CPU_ATTENTION_WITNESS_AUDIT.

GPU reproduction uses the original external checkpoint/tokenizer and fixed hashed loader. In a fresh isolated root adapt filesystem/device paths and establish a new relative180min clock, preserving inputs/operator/statisticalgates. Run prepare.py (for exact frozen fixture), run_A.py technical, CPU witness audit, run_A.py development, analyze_A.py development, run_A.py fresh_confirmation, analyze_A.py fresh_confirmation. B/C are gated and must not be claimed executed when the prerequisites fail. Same-variable and zero full-vocabulary identity gates must pass before effects are interpreted.

Main row operation is originalSDPA+(FP64counterrow-FP64baserow) at one target; original projection/normalization/residual/MLP remain. Key swaps alter normalization across the entire causal row, not just a single weight. Equal-dose shuffles target the actual BF16 row correction; all unmatched cases are retained. No10%cap from old CATC is applied.
''');print('EDGE_PACKAGE_READY')
