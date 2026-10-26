"""Synthetic in-memory contract fixtures; never GPU measurements/predictions."""
import copy
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import PROTOCOL_SHA
from audit_actual_halting import (CATEGORIES, METRICS, METHODS, producer_fingerprint,
                                  selected_settings, validate_records)
from policies import exit_index

def fixture(model, layers):
    documents, offline = [], {}
    for task in CATEGORIES:
        for index in range(10):
            uid = f'unit-only/{task}/{index}'
            doc = dict(id=uid, task=task, choices=['a', 'bb', 'ccc', 'dddd'],
                       request_pairs=[['context', str(x)] for x in range(4)],
                       choice_char_lengths=[1, 2, 3, 4], gold=index % 4,
                       primary_metric='acc' if task == 'mmlu' else 'acc_norm')
            documents.append(doc)
            trajectory, predictions = [], {}
            for depth in range(1, 5):
                trajectory.append(dict(depth=depth,
                    full_vocab_kl=[None, .5, .08, .01][depth-1],
                    hidden_relative_l2=[None, .8, .4, .1][depth-1],
                    native_gate_cdf=[.125, .625, .875, 1.][depth-1]))
                raw = np.array([-2., -3., -4., -5.])
                raw[(index+depth) % 4] = -.125
                normalized = raw/np.array(doc['choice_char_lengths'])
                chosen = int((raw if doc['primary_metric'] == 'acc' else normalized).argmax())
                predictions[f'r{depth}'] = dict(sum_scores=raw.tolist(),
                    normalized_scores=normalized.tolist(), prediction=chosen,
                    raw_prediction=int(raw.argmax()), normalized_prediction=int(normalized.argmax()),
                    correct=chosen == doc['gold'], primary_metric=doc['primary_metric'],
                    gold_nll_per_token=float(-raw[doc['gold']]/(doc['gold']+1)))
            offline[uid] = dict(trajectory=trajectory, predictions=predictions)
    selection = {metric: {task: {'3': threshold} for task in CATEGORIES}
                 for metric, threshold in zip(METRICS, [.1, .2, .6])}
    settings = selected_settings(selection)
    measured, cases = [], []
    for doc in documents:
        uid = doc['id']
        for method in METHODS:
            threshold = None if method.startswith('fixed') else settings[method][doc['task']]['threshold']
            depth = int(method[-1]) if method.startswith('fixed') else exit_index(offline[uid]['trajectory'], method, threshold)+1
            reference = offline[uid]['predictions'][f'r{depth}']
            history = [] if method.startswith('fixed') else [dict(depth=i+1, metric=method,
                        value=offline[uid]['trajectory'][i][method]) for i in range(depth)]
            cases.append(dict(id=uid, method=method, depth=depth, passed=True,
                              max_score_error=0., actual_decoder_layer_calls=4*layers*depth))
            for repeat in range(5):
                measured.append(dict(record_key=f'{uid}|{method}|{repeat}', id=uid,
                    task=doc['task'], model=model, method=method, repeat=repeat,
                    protocol_sha256=PROTOCOL_SHA, implementation_sha256=producer_fingerprint(),
                    depth=depth, threshold=threshold, expected_decoder_layer_calls=4*layers*depth,
                    actual_decoder_layer_calls=None, elapsed_s=.5+repeat*.01,
                    peak_memory_bytes=1024, sum_scores=list(reference['sum_scores']),
                    token_counts=[1, 2, 3, 4], result=copy.deepcopy(reference),
                    history=copy.deepcopy(history), measured_repeat_max_score_error=0.,
                    measured_repeat_quality_matches_offline=True))
    return dict(model=model, layers=layers, measured=measured,
                parity=dict(passed=True, cases=cases), documents=documents,
                offline=offline, selection=selection, stored_settings=settings)

class ActualHaltingAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixtures = [fixture('Ouro-1.4B', 24), fixture('Ouro-2.6B', 48)]

    def rejects(self, mutate):
        for source in self.fixtures:
            with self.subTest(model=source['model']):
                value = copy.deepcopy(source)
                mutate(value)
                with self.assertRaises(AssertionError):
                    validate_records(**value)

    def test_consistent_synthetic_contract(self):
        for value in self.fixtures:
            result = validate_records(**value)
            self.assertEqual(result['timed_records'], 750)
            self.assertEqual(result['layer_count_parity_cases'], 150)

    def test_duplicate_cannot_replace_missing_record(self):
        self.rejects(lambda v: v['measured'].__setitem__(0, copy.deepcopy(v['measured'][1])))

    def test_extra_question_cannot_replace_frozen_id(self):
        def mutate(value):
            row = value['measured'][0]
            row['id'] = 'unit-only/unselected/999'
            row['record_key'] = f"{row['id']}|{row['method']}|{row['repeat']}"
        self.rejects(mutate)

    def test_wrong_observed_layer_count(self):
        def mutate(value):
            value['parity']['cases'][0]['actual_decoder_layer_calls'] += value['layers']
        self.rejects(mutate)

    def test_wrong_exit_even_with_consistent_claimed_work(self):
        def mutate(value):
            row = value['measured'][0]
            row['depth'] = 3
            row['expected_decoder_layer_calls'] = 4*value['layers']*3
        self.rejects(mutate)

    def test_score_error_recomputed_instead_of_trusting_flag(self):
        def mutate(value):
            row = value['measured'][0]
            row['sum_scores'][0] += .05
            row['result']['sum_scores'][0] += .05
        self.rejects(mutate)

    def test_future_depth_in_history(self):
        def mutate(value):
            row = next(r for r in value['measured'] if r['method'] == 'full_vocab_kl')
            row['history'][0]['depth'] = 4
        self.rejects(mutate)

    def test_quality_boolean_does_not_substitute_for_prediction(self):
        def mutate(value):
            row = value['measured'][0]
            row['result']['prediction'] = (row['result']['prediction']+1) % 4
        self.rejects(mutate)

    def test_stored_settings_must_equal_frozen_validation_choice(self):
        def mutate(value):
            value['stored_settings']['native_gate_cdf']['mmlu']['threshold'] += .01
        self.rejects(mutate)

if __name__ == '__main__':
    unittest.main()
