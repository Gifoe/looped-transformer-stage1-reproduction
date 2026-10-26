"""Semantic checks for the actual causal stop decisions, no model download."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from policies import bf16_scalar,exit_index,exits

class PolicyTests(unittest.TestCase):
    def test_native_bfloat_rounding_at_boundary(self):
        self.assertEqual(bf16_scalar(.45),.44921875)
        self.assertTrue(exits(.44921875,'native_gate_cdf',.45))
        self.assertFalse(exits(.447265625,'native_gate_cdf',.45))

    def test_future_states_cannot_change_a_stop(self):
        past=[{'full_vocab_kl':None},{'full_vocab_kl':.01}]
        self.assertEqual(exit_index(past+[{'full_vocab_kl':999},{'full_vocab_kl':999}],'full_vocab_kl',.02),1)
        self.assertEqual(exit_index(past+[{'full_vocab_kl':0},{'full_vocab_kl':0}],'full_vocab_kl',.02),1)

    def test_missing_first_transition_is_not_zero(self):
        self.assertEqual(exit_index([{'hidden_relative_l2':None},{'hidden_relative_l2':3},{'hidden_relative_l2':2},{'hidden_relative_l2':1}],'hidden_relative_l2',.1),3)

if __name__=='__main__':unittest.main()
