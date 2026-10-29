import copy,sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analysis_core import features,fixed_policies,ACTIONS,ALL,paired
def fixture():
 rng=np.random.default_rng(45)
 views={}
 for v in ['official','format']:
  views[v]=dict(sum_scores={a:rng.normal(-8,1,4).tolist() for a in ALL},token_logprobs={a:rng.normal(-2,1,(4,3)).tolist() for a in ALL},probe_nll={a:float(rng.uniform(1,4)) for a in ALL},geometry=dict(align23=.1,align34=.4,align24=.2,curvature=.8,persistence=.1,norm2=1,norm3=2,norm4=3),update_projections=rng.normal(size=96).tolist(),controls=dict(confidence=.4,entropy=2,vocab_margin=.1,kl41=.3))
 for v in views.values():v['probe']=dict(prefix_tokens=10)
 return dict(id='x',gold=0,choice_char_lengths=[9,12,4,5],views=views)
class Tests(unittest.TestCase):
 def test_gold_and_id_do_not_enter_features_or_policy(self):
  r=fixture();s=copy.deepcopy(r);s['gold']=3;s['id']='other';s['task']='sciq'
  for kind in ['controls','new','combined']:np.testing.assert_array_equal(features(r,kind)[0],features(s,kind)[0])
  self.assertEqual(fixed_policies(r),fixed_policies(s))
 def test_candidate_permutation_invariance(self):
  r=fixture();s=copy.deepcopy(r);perm=[2,0,3,1];s['choice_char_lengths']=[r['choice_char_lengths'][i] for i in perm]
  for v in s['views']:
   for key in ['sum_scores','token_logprobs']:
    for a in ALL:s['views'][v][key][a]=[r['views'][v][key][a][i] for i in perm]
  for kind in ['controls','new','combined']:np.testing.assert_allclose(features(r,kind)[0],features(s,kind)[0],atol=1e-12)
  p,a=fixed_policies(r);q,b=fixed_policies(s);self.assertEqual(a,b)
  self.assertTrue(all(perm[q[k]]==p[k] for k in p))
 def test_paired_net(self):
  out=paired([1,0,1,1],[0,1,0,1]);self.assertEqual((out['corrections'],out['degradations'],out['net']),(2,1,1))
 def test_absent_probe_defaults_without_fabricating_loss(self):
  r=fixture();r['views']['official']['probe_nll']={a:None for a in ALL}
  self.assertEqual(fixed_policies(r)[1]['proxy'],'adaptive')
  values,names=features(r,'new');self.assertEqual(values[names.index('probe_available')],0)
  self.assertTrue(np.isfinite(values).all())
if __name__=='__main__':unittest.main()
