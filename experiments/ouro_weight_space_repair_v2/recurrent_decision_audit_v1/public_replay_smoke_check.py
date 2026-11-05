"""Execute the public-only finalize path with no private transport receipts."""
from pathlib import Path
import json,tempfile,shutil,importlib.util,hashlib
O=Path(__file__).resolve().parent
names=['permutation_stability.csv','k1_k4_paired_transitions.csv','stability_summary.json','paired_statistics.json','official_environment_exact_replay.json','runtime_metadata.json','a4_SERVER_PREFLIGHT_PUBLIC.json','a4_START_RECEIPT.json','a4_OFFICIAL_ANALYZER_RECEIPT.json','live_resource_cache_preflight.json','existing_score_parity.json','compute_ledger.json','a4_gpu_budget.json','depth_readout_coverage.json','loopwise_predictions.csv']
def main():
 with tempfile.TemporaryDirectory(prefix='public_replay_',dir=O) as tmp:
  base=Path(tmp).resolve();assert base.is_relative_to(O.resolve());a=base/O.name;a.mkdir();(base/'results').mkdir();shutil.copyfile(O.parent/'results/r33_RESOURCE_CACHE_PREFLIGHT_PUBLIC.json',base/'results/r33_RESOURCE_CACHE_PREFLIGHT_PUBLIC.json')
  for n in names:shutil.copyfile(O/n,a/n)
  assert not any(a.glob('*RAW*')) and not (a/'official_server_cpu_replay.json').exists()
  spec=importlib.util.spec_from_file_location('replay_finalize',O/'finalize_cpu.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.O=a;m.main()
  for n in ['stability_correctness_cross_tab.json','compute_ledger.json','gold_margin_descriptive_summary.json']:
   assert json.loads((a/n).read_text(encoding='utf8'))==json.loads((O/n).read_text(encoding='utf8'))
 result=dict(status='passed_public_only_replay_no_private_receipts',source_sha256=hashlib.sha256((O/'finalize_cpu.py').read_bytes()).hexdigest(),checked_outputs=['stability_correctness_cross_tab.json','compute_ledger.json','gold_margin_descriptive_summary.json'],generated_figures=True,model_or_GPU_loaded=False)
 (O/'PUBLIC_REPLAY_SMOKE_CHECK.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
