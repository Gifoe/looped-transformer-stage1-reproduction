import hashlib
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import CFG,read

class FrozenProtocolTest(unittest.TestCase):
    def test_new_data_disjoint_from_stage1(self):
        current={name:read('data/'+name+'.jsonl') for name in ['validation','qwen_test','ouro_test','math_validation','math_test','math_fewshot']}
        for rows in current.values():self.assertEqual(len(rows),len({r['id'] for r in rows}))
        old=read(BASE/'data/validation.jsonl')+read(BASE/'data/qwen_test.jsonl')
        old_ids={r['id'] for r in old}
        self.assertFalse(old_ids&{r['id'] for name in ['validation','qwen_test'] for r in current[name]})
        self.assertEqual(len(current['validation']),250);self.assertEqual(len(current['qwen_test']),600);self.assertEqual(len(current['ouro_test']),300)
        self.assertLessEqual({r['id'] for r in current['ouro_test']},{r['id'] for r in current['qwen_test']})
        old_math=read(BASE/'data/math_test.jsonl')
        self.assertFalse({r['id'] for r in old_math}&{r['id'] for r in current['math_test']})
        self.assertEqual(len(current['math_validation']),100);self.assertEqual(len(current['math_test']),200)
        groups=[{ ' '.join(r['question'].casefold().split()) for r in current[name]} for name in ['math_validation','math_test','math_fewshot']]
        for i in range(3):
            for j in range(i):self.assertFalse(groups[i]&groups[j])

    def test_requests_and_prompt_identity(self):
        for name in ['validation','qwen_test']:
            for row in read('data/'+name+'.jsonl'):
                category='mmlu' if row['task'].startswith('mmlu_') else row['task']
                self.assertEqual(row['fewshot_count'],CFG['fewshot'][category])
                self.assertEqual(hashlib.sha256(row['prompt'].encode()).hexdigest(),row['prompt_sha256'])
                self.assertTrue(all(pair[0]==row['prompt'] for pair in row['request_pairs']))
                self.assertEqual(len(row['request_pairs']),len(row['choices']))
                self.assertEqual(len(row['choice_char_lengths']),len(row['choices']))
                self.assertTrue(all(n>0 for n in row['choice_char_lengths']))

    def test_frozen_manifest_hashes(self):
        audit=json.loads((ROOT/'data/manifest_audit.json').read_text())
        for name,record in audit['manifests'].items():
            self.assertEqual(hashlib.sha256((ROOT/'data'/(name+'.jsonl')).read_bytes()).hexdigest(),record['sha256'])

    def test_original_stage1_files_unchanged(self):
        manifest=json.loads((BASE/'ARTIFACT_MANIFEST.json').read_text())
        for name,record in manifest.items():
            self.assertEqual(hashlib.sha256((BASE/name).read_bytes()).hexdigest(),record['sha256'],name)

if __name__=='__main__':unittest.main()
