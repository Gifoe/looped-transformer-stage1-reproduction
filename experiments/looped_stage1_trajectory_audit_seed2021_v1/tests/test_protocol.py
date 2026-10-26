import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from analyze import paired_ci,policy_index,selections,trajectory_pattern_masks

class ProtocolTest(unittest.TestCase):
    def test_oscillation_ends_at_final_depth(self):
        masks=trajectory_pattern_masks([[True,False,True,False],[False,True,False,True]])
        self.assertEqual(masks['correct_wrong_correct'].tolist(),[False,True])
        self.assertEqual(masks['wrong_correct_wrong'].tolist(),[True,False])

    def test_halting_ignores_gold_and_future(self):
        trajectory=[dict(choice_confidence=.9,correct=False),dict(choice_confidence=.5,correct=True),dict(choice_confidence=.7,correct=False),dict(choice_confidence=.8,correct=True)]
        self.assertEqual(policy_index(trajectory,'choice_confidence',.8),0)
        altered=[dict(choice_confidence=.9,correct=True),dict(choice_confidence=.99,correct=False)]
        self.assertEqual(policy_index(altered,'choice_confidence',.8),0)

    def test_kl_cannot_exit_before_second_state(self):
        trajectory=[dict(full_vocab_kl=None),dict(full_vocab_kl=.001),dict(full_vocab_kl=.01),dict(full_vocab_kl=.001)]
        self.assertEqual(policy_index(trajectory,'full_vocab_kl',.005),1)
        self.assertEqual(policy_index(trajectory,'full_vocab_kl',0),3)

    def test_paired_bootstrap_preserves_pairing(self):
        identical=paired_ci([1,0,1,0],[1,0,1,0])
        self.assertEqual(identical,dict(delta=0,ci_low=0,ci_high=0))
        self.assertEqual(paired_ci([0]*8,[1]*8),dict(delta=1,ci_low=1,ci_high=1))

    def test_manifest_splits_and_alignment(self):
        path=ROOT/'data/manifest_audit.json'
        if not path.exists():self.skipTest('Immutable datasets not acquired yet')
        ids={}
        for split in ['validation','ouro_test','qwen_test']:
            rows=[json.loads(l) for l in (ROOT/'data'/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
            ids[split]={r['id'] for r in rows}
            self.assertEqual(len(ids[split]),len(rows))
            for r in rows:self.assertTrue(0<=r['gold']<len(r['choices']))
        self.assertFalse(ids['validation']&ids['qwen_test'])
        self.assertLessEqual(ids['ouro_test'],ids['qwen_test'])

if __name__=='__main__':unittest.main()
