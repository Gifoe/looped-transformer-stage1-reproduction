import json
import time
import traceback
from run_pipeline import ROOT,run,status

def main():
    while True:
        state=json.loads((ROOT/'STATUS.json').read_text())
        if state.get('stage')=='all_available_experiments_finished':break
        if state.get('state')=='failed':raise RuntimeError('Prior stage failed: '+str(state))
        time.sleep(5)
    run('math_depth_and_hidden',['scripts/math_depth_audit.py'],gpu=True)
    run('scoring_and_timing',['scripts/scoring_and_timing.py'],gpu=True)
    run('final_protocol_tests',['-m','unittest','discover','-s','tests','-v'])
    run('final_statistics_rebuild',['scripts/rebuild_all.py'])
    run('research_report',['scripts/write_findings.py'])
    run('artifact_package',['scripts/package_artifacts.py'])
    status(stage='completed',state='completed_with_scope_limitations',author_patch_parity='unavailable',decision='No-Go')

if __name__=='__main__':
    try:main()
    except Exception as exc:status(stage='finalization_failed',state='failed',error=str(exc));traceback.print_exc();raise
