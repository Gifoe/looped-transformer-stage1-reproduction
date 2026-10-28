"""Leakage, neutral-update handling, split boundaries and paired-count contracts."""
from pathlib import Path
import json
import sys
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from quality_core import BASE,PLAN,read,features,visible,predict_ridge


class QualityContracts(unittest.TestCase):
    def test_id_join_survives_reordered_euler(self):
        a=read(BASE/'results/qwen_A0_test.jsonl');b=read(BASE/'results/qwen_A3_test.jsonl')[::-1]
        by_id={r['id']:r for r in b}
        gain=[int(by_id[r['id']]['correct'])-int(r['correct']) for r in a]
        self.assertEqual(gain.count(1),32);self.assertEqual(gain.count(-1),15)
        self.assertNotEqual([r['id'] for r in a],[r['id'] for r in b])

    def test_gold_and_evaluation_fields_cannot_change_features(self):
        a=read(BASE/'results/qwen_A0_test.jsonl')[0];b=read(BASE/'results/qwen_A3_test.jsonl')[0]
        expected=features(visible(a),visible(b))
        for r in [a,b]:r.update(gold=999,correct='FORBIDDEN',gold_nll_per_token=-1e50,gold_choice_logp=1e99,oracle_action='accept')
        self.assertEqual(expected,features(visible(a),visible(b)))
        self.assertFalse(set(expected)&{'gold','task','gain','correct','oracle_action'})

    def test_every_policy_preserves_all600(self):
        f=pd.DataFrame(read(ROOT/'results/policy_predictions.jsonl'))
        for (scope,policy),g in f.groupby(['scope','policy']):
            self.assertEqual(len(g),600,(scope,policy));self.assertEqual(g.id.nunique(),600)
            self.assertEqual(int(g.changed.sum()),75)
            self.assertEqual(int(((g.gain==0)&g.changed).sum()),28)

    def test_nested_outer_and_inner_boundaries_and_train_scaling(self):
        folds=json.loads((ROOT/'data/outer_folds.json').read_text())
        feats=pd.read_csv(ROOT/'data/discovery_features.csv').set_index('id')
        labels=pd.read_csv(ROOT/'data/discovery_labels.csv').set_index('id')
        models=json.loads((ROOT/'results/fitted_models_and_selection.json').read_text())
        for fit in models:
            if fit['scope']!='nested_cv':continue
            fold=folds[fit['fold']-1];tr=set(fit['training_ids']);te=set(fold['test_ids'])
            self.assertFalse(tr&te);self.assertTrue(tr<=set(fold['train_ids']))
            if fit['policy'] in ['linear','original_only_linear']:
                all_changed=set(i for i in fold['train_ids'] if bool(labels.loc[i,'changed']))
                self.assertEqual(tr,all_changed)
                X=feats.loc[fit['training_ids'],fit['features']].to_numpy(float)
                np.testing.assert_allclose(np.mean(X,axis=0),fit['model']['mean'],rtol=1e-12,atol=1e-12)
                for inner in fit['model']['inner_folds']:
                    # Stored indices index the original600-row data, never held-out outer rows.
                    ids=list(labels.index)
                    it={ids[i] for i in inner['train']};iv={ids[i] for i in inner['validation']}
                    self.assertFalse(it&iv);self.assertTrue((it|iv)<=tr)

    def test_random_gate_accept_counts_and_gain_identity(self):
        f=pd.DataFrame(read(ROOT/'results/policy_predictions.jsonl'))
        for scope in f.scope.unique():
            for task in f.group.unique():
                x=f[(f.scope==scope)&(f.group==task)]
                for name in ['confidence_gate','margin_entropy_gate','kl_gate','linear']:
                    self.assertEqual(int(x[x.policy==name].accept_euler.sum()),int(x[x.policy=='matched_random_'+name].accept_euler.sum()))
        s=pd.read_csv(ROOT/'results/policy_summary.csv')
        for _,r in s.iterrows():self.assertEqual(r.net_gain_over_euler,r.harmful_avoided-r.beneficial_rejected)

    def test_cross_dataset_models_never_see_target(self):
        f=pd.DataFrame(read(ROOT/'results/policy_predictions.jsonl'))
        labels=pd.read_csv(ROOT/'data/discovery_labels.csv').set_index('id')
        for fit in json.loads((ROOT/'results/fitted_models_and_selection.json').read_text()):
            if fit['scope']!='cross_dataset':continue
            held=f[(f.scope=='cross_dataset')&(f.fold==fit['fold'])]
            self.assertEqual(held.group.nunique(),1)
            target=held.group.iloc[0]
            self.assertTrue(all(labels.loc[i,'group']!=target for i in fit['training_ids']))


if __name__=='__main__':unittest.main(verbosity=2)
