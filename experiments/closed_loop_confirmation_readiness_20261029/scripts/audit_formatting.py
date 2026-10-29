"""Synthetic template parity only: no benchmark examples, labels, fitting or model calls."""
from pathlib import Path
import hashlib
import json

import yaml
from jinja2 import Template

from format_observation import arc_easy, commonsense_qa

ROOT = Path(__file__).resolve().parents[1]


def main():
    provenance = json.loads((ROOT / 'sources/TASK_CONFIG_PROVENANCE.json').read_text())
    for rel, digest in provenance['files'].items():
        assert hashlib.sha256((ROOT / 'sources/installed_lm_eval' / rel).read_bytes()).hexdigest() == digest
    n = 0
    for question in ['Synthetic question?', '  Synthetic question with boundary spaces?  ', 'Synthetic Unicode question: α β?', 'Synthetic multiline\nquestion?']:
        for task, formatter, labels, texts in [
            ('arc/arc_easy.yaml', arc_easy, ['1', '2', '3'], ['short', 'a longer choice', 'Unicode α']),
            ('commonsense_qa/default.yaml', commonsense_qa, list('ABCDE'), ['first', 'second', 'third', 'fourth', 'fifth']),
        ]:
            config = yaml.safe_load((ROOT / 'sources/installed_lm_eval' / task).read_text())
            document = {'question': question, 'choices': {'label': labels, 'text': texts}}
            prompt = Template(config['doc_to_text']).render(**document)
            obs = formatter(question, labels, texts)
            candidates = texts if obs['task'] == 'arc_easy' else labels
            assert obs['variants']['official'] == [[prompt, ' ' + choice] for choice in candidates]
            assert obs['choice_char_lengths'] == [len(choice) for choice in candidates]
            assert obs['question'] in prompt and 'gold' not in obs and 'answerKey' not in obs
            if obs['task'] == 'commonsense_qa':
                assert obs['primary_metric'] == 'acc' and all(len(answer) == 2 for _, answer in obs['variants']['official'])
            n += 1
    result = {'scope': 'Synthetic label-free prompt/continuation/length parity with installed task YAML; not tokenizer/logit parity or benchmark evaluation', 'synthetic_cases': n, 'passed': True, 'benchmark_examples_accessed': False, 'model_calls': 0, 'gpu_seconds': 0}
    (ROOT / 'results').mkdir(exist_ok=True)
    (ROOT / 'results/SYNTHETIC_FORMAT_AUDIT.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
