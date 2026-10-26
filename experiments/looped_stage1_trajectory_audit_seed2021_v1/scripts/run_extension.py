"""Sequential continuation; waits for core analysis before any new GPU process."""
import json
from pathlib import Path
import time
import traceback
from run_pipeline import ROOT,run,status

def main():
    while not (ROOT/'configs/validation_selected.json').exists():
        state=json.loads((ROOT/'STATUS.json').read_text())
        if state.get('state')=='failed':raise RuntimeError('Core pipeline failed: '+str(state))
        time.sleep(5)
    while not (ROOT/'results/halting_curves.csv').exists():
        state=json.loads((ROOT/'STATUS.json').read_text())
        if state.get('stage')=='ouro_core_analysis_finished':break
        if state.get('state')=='failed':raise RuntimeError('Core pipeline failed: '+str(state))
        time.sleep(5)
    run('qwen_independent_recipe',['scripts/qwen_audit.py'],gpu=True)
    run('qwen_analysis',['scripts/analyze_qwen.py'])
    run('continuation_diagnostics',['scripts/additional_diagnostics.py'])
    run('math_generation',['scripts/math_generation.py'],gpu=True)
    status(stage='all_available_experiments_finished',state='partial',official_source_parity='unavailable: author repository 404')

if __name__=='__main__':
    try:main()
    except Exception as exc:status(stage='extension_failed',state='failed',error=str(exc));traceback.print_exc();raise
