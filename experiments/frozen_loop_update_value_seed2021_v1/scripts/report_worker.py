"""Use the audited server dependency paths/CPU affinity for CPU report scripts."""
import runtime_bootstrap
import sys,runpy
from pathlib import Path
name=sys.argv[1]
assert name in ['diagnose_signals.py','render_report.py','render_appendix.py']
runpy.run_path(str(Path(__file__).parent/name),run_name='__main__')
