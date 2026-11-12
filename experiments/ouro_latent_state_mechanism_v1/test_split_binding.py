"""CPU regression: target split and validation split must both be hash-bound."""
import inspect
import pyarrow.parquet as pq
import cohort_runner as runner
import gpu_parity_gate as gate
expected={'train':1119,'validation':299,'test':1172}
for split,n in expected.items():
 path=runner.split_data_path(split);assert len(pq.read_table(path))==n
 hashes=runner.bound_hashes(split)
 for name in {split,'validation'}:
  p=runner.split_data_path(name);assert hashes[str(p.relative_to(runner.PROJECT))]==runner.sha(p)
assert "split_data_path('validation')" in inspect.getsource(gate.main)
assert 'execute_request(model,enc)' in inspect.getsource(gate.main)
print('PASS split binding includes validation and target; gate calls production request')
