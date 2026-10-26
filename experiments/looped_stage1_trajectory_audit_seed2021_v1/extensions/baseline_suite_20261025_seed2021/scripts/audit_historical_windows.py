"""Verify historical window metrics without changing the Stage1 package."""
from datetime import datetime,timezone
import hashlib
import math
from common import ROOT,BASE,read,save

def main():
    baseline={r['id']:r for r in read(BASE/'results/qwen_A0_test.jsonl')};assert len(baseline)==600
    evidence=[]
    for method,iterations in [('A1',2),('A2',6),('A3',6),('A4',6)]:
        path=BASE/f'results/qwen_{method}_test.jsonl';rows=read(path)
        assert len(rows)==600 and {r['id'] for r in rows}==set(baseline)
        for row in rows:
            assert row['gold']==baseline[row['id']]['gold']
            window=row['window'];assert window and len(window['steps'])==iterations
            assert all(math.isfinite(window[name][metric]) for name in ['input_to_output','input_to_natural'] for metric in ['relative_l2','cosine'])
            assert all(math.isfinite(s['relative_l2']) and math.isfinite(s['cosine']) for s in window['steps'])
        evidence.append(dict(method=method,n=600,iterations=iterations,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),all_window_metrics_present_finite=True,mean_input_output_relative_l2=sum(r['window']['input_to_output']['relative_l2'] for r in rows)/600,mean_input_output_cosine=sum(r['window']['input_to_output']['cosine'] for r in rows)/600))
    save('results/historical_qwen_window_audit.json',dict(passed=True,checked_utc=datetime.now(timezone.utc).isoformat(),source='sealed Stage1 zero-shot dataset; separate from new few-shot suite',configurations=evidence))
    print('Historical Qwen windows verified: four configurations x 600 aligned questions',flush=True)

if __name__=='__main__':main()
