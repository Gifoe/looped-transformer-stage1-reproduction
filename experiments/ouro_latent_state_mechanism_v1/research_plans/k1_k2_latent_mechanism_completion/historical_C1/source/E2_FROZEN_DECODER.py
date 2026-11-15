"""Prediction-only adapter for the original E2 v3 frozen diagnostic decoders.

No fit, hyperparameter choice, data extraction, evaluation or GPU work occurs at
import. These are supervised diagnostic heads, not a training-free controller.
"""
import hashlib
from pathlib import Path
import numpy as np

ORIGINAL_NPZ_SHA256='204a3788301c7aa826a198b9e40ce495ef81284b6cd41a4758dfd942f8650890'
GLOBAL_TOKEN_IDS=(330,340,389,414,422)  # Train-derived sorted A,C,B,E,D columns.
MODEL_KEYS=tuple(f'{site}_{feature}_trainK{loop}_{control}'
    for site in ['l18_input','post_norm'] for feature in ['raw','unit_l2']
    for loop in [1,2] for control in ['real','permuted'])

class FrozenE2Decoders:
    def __init__(self,path):
        self.path=Path(path)
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        assert self.sha256==ORIGINAL_NPZ_SHA256,'Requires the exact original v3 decoder/scaler artifact'
        with np.load(self.path,allow_pickle=False) as z:
            assert set(z.files)=={key+suffix for key in MODEL_KEYS for suffix in ['_w','_mu','_std']}
            self.arrays={key:z[key].copy() for key in z.files}
        for key in MODEL_KEYS:
            w,mu,std=[self.arrays[key+suffix] for suffix in ['_w','_mu','_std']]
            assert w.shape==(2049,5) and mu.shape==std.shape==(2048,)
            assert all(v.dtype==np.float64 and np.isfinite(v).all() for v in [w,mu,std])
            assert (std>0).all()
            for value in [w,mu,std]:value.setflags(write=False)

    def predict(self,states,key,offered_token_ids):
        """Apply one fixed head to any supplied rows; no gold or loop label input.

        Caller supplies original-coordinate R vectors at the head's documented
        site, BF16 converted via FP32 then FP64. Different sites/hardware retain
        their provenance; this adapter does not establish transfer validity.
        """
        assert key in MODEL_KEYS
        x=np.asarray(states,dtype=np.float64)
        if x.ndim==1:x=x[None,:]
        assert x.ndim==2 and x.shape[1]==2048 and np.isfinite(x).all()
        offered=tuple(int(v) for v in offered_token_ids)
        assert len(offered)>=2 and len(set(offered))==len(offered)
        assert set(offered)<=set(GLOBAL_TOKEN_IDS),'Unsupported offered label vocabulary'
        if '_unit_l2_' in key:
            norms=np.linalg.norm(x,axis=1,keepdims=True)
            x=np.divide(x,norms,out=np.zeros_like(x),where=norms!=0)
        mu=self.arrays[key+'_mu'];std=self.arrays[key+'_std'];w=self.arrays[key+'_w']
        features=(x-mu)/std
        augmented=np.concatenate([features,np.ones((len(x),1),dtype=np.float64)],axis=1)
        global_scores=augmented@w
        columns=[GLOBAL_TOKEN_IDS.index(v) for v in offered]
        scores=global_scores[:,columns]
        assert np.isfinite(scores).all()
        # Preserve the original v3 global-column mask and reduction order.
        from scipy.special import logsumexp
        mask=np.zeros_like(global_scores,dtype=bool);mask[:,columns]=True
        masked=np.where(mask,global_scores,-np.inf)
        global_prob=np.exp(masked-logsumexp(masked,axis=1,keepdims=True))
        prob=global_prob[:,columns]
        return {'diagnostic_model_key':key,'parameter_sha256':self.sha256,
                'offered_token_ids':list(offered),'scores':scores,'probabilities':prob,
                'prediction_offered_index':scores.argmax(axis=1),
                'top_ties_offered_indices':[np.flatnonzero(row==row.max()).tolist() for row in scores],
                'scope':'Original frozen supervised E2 diagnostic head. No new fit or native causal-use claim.'}
