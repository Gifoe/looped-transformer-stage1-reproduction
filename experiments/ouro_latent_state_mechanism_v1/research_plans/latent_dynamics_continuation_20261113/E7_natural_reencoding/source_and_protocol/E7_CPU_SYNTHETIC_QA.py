"""Synthetic mathematical controls only. Does not open any E7 outcome or label file."""
from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
root=Path(__file__).parent;s=importlib.util.spec_from_file_location('analysis',root/'E7_REENCODING_ANALYSIS.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
x=np.zeros((4,4));x[1:,:]=1;assert np.array_equal(a.contrasts(x),[1,0,0])
x=np.zeros((4,4));x[:,1:]=1;assert np.array_equal(a.contrasts(x),[0,1,0])
x=np.zeros((4,4));x[1:,1:]=1;assert np.array_equal(a.contrasts(x),[.75,.75,1])
assert np.array_equal(a.contrasts(np.ones((3,4))),[0,0,0]);assert np.array_equal(a.contrasts(np.ones((5,4))),[0,0,0])
one=a.intervals(np.array([[1,0,0],[0,1,0],[-1,0,1]]),['a','b','c']);two=a.intervals(np.array([[1,0,0],[0,1,0],[-1,0,1]]),['a','b','c']);assert one==two and all(r['bootstrap_questions']==10000 for r in one)
m=a.import_methods('/root/autodl-tmp/ouro-e6a/E6A/E6A_METHODS.py');q={'id':'SYNTHETIC','split':'train','gold':1,'m':4,'views':{}}
for p in range(4):
 for ss in range(4):
  v=np.array([[-4.,-2.,-3.,-5.]]*4);q['views'][p,ss]={'scores':v+(100. if p==2 else 0.),'token_count':12,'replica':{'loopcd_adaptive_cap1':{'prediction_semantic':1}},'pred':np.ones(4,int)}
r=a.methods_for(q,m);assert r['fixed_pair_K2_K2']['prediction']==1 and np.array_equal(r['fixed_pair_K2_K2']['scores_semantic'],[-.5,1.5,.5,-1.5]);q['views'][2,0]['scores'][:]=0;q['views'][0,0]['scores'][:]=0;r=a.methods_for(q,m);assert r['fixed_pair_K2_K2']['prediction']==0 and r['fixed_pair_K2_K2']['top_semantic_set']==[0,1,2,3]
receipt={'status':'PASS','scope':'Only hand-constructed data, no newvariant scientific outcome/labels read','pure_P':[1,0,0],'pure_S':[0,1,0],'interaction_grid':[.75,.75,1],'extensions_rectangular_grid':True,'bootstrap_seed_replay_exact':True,'unique_pair_FP64_centered_constant_shift_and_tie_rule':True,'source_sha256':hashlib.sha256((root/'E7_REENCODING_ANALYSIS.py').read_bytes()).hexdigest()};(root/'E7_CPU_SYNTHETIC_QA.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
