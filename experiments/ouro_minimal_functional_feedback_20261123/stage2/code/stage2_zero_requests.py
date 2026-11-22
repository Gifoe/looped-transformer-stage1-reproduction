"""Export zero-shot requests with the same pinned Harness task definitions."""
from pathlib import Path
import sys

import stage2_requests


if __name__ == "__main__":
    stage2_requests.EVALUATIONS = {
        task: (0, repo, evaluation, None)
        for task, (_, repo, evaluation, _) in stage2_requests.EVALUATIONS.items()
    }
    stage2_requests.PAPER = None
    if len(sys.argv) > 1 and sys.argv[1] == "--canonical-mmlu":
        del sys.argv[1]
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recovery"))
        import mmlu_recovery

        mmlu_recovery.main()
    else:
        stage2_requests.main()
