"""Aggregate measured forward windows without mislabeling the final reset window as a run peak."""
import argparse,json,datetime
from pathlib import Path
a=argparse.ArgumentParser();a.add_argument('results');args=a.parse_args();p=Path(args.results);rows=[]
for f in sorted(p.glob('*.native.json')):rows.extend(json.loads(f.read_text())['native'].values())
for f in sorted(p.glob('*.results.json')):
 obj=json.loads(f.read_text());rows.extend(obj['rows'])
 if obj.get('technical_reused_kv_full_forward_bridge'):rows.append(obj['technical_reused_kv_full_forward_bridge'])
assert rows
receipt=json.loads((p/'receipt.json').read_text());launch=(p/'launch_utc.txt').read_text().strip();finish=datetime.datetime.fromtimestamp((p/'receipt.json').stat().st_mtime,datetime.timezone.utc);start=datetime.datetime.fromisoformat(launch.replace('Z','+00:00'))
r={'scope':'MAX_OVER_MEASURED_FORWARD_WINDOWS_EXCLUDES_MODEL_LOADING','n_forward_windows':len(rows),'forward_wall_seconds_sum':sum(x['wall_seconds'] for x in rows),'max_peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in rows),'max_peak_reserved_bytes':max(x['peak_reserved_bytes'] for x in rows),'max_cpu_rss_bytes':max(x['cpu_rss_bytes'] for x in rows),'process_cpu_maxrss_kib':max(x['cpu_maxrss_kib'] for x in rows),'collector_receipt_wall_seconds':receipt['wall_seconds'],'launch_utc':launch,'receipt_written_utc_from_filesystem':finish.isoformat(),'command_window_seconds_until_receipt':(finish-start).total_seconds(),'command_window_precision_note':'Launch timestamp has whole-second precision; receipt time derives from its filesystem mtime. Collector timer starts after preflight file hashing.'}
r['GPU_measurement_exclusions']=['model_loading','between_forward_hybrid_derivation','between_forward_control_construction','tensor_serialization_outside_run']
r['GPU_scope_note']='Recorded forward-window maxima only, not a full-collector GPU memory peak. CPU maxrss is process-wide.'
(p/'resource_summary.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
