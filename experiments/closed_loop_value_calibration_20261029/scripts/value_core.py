"""Strict observation-only inputs for the tiny action-value calibrators."""
import numpy as np
ALPHAS=['0.0','-1.0','-0.5','0.5']
KINDS=['input','confidence','combined']
def observation(row):return {'decision':row['decision'],'controls':row['controls']}
def features(obs,kind,action):
    assert set(obs)=={'decision','controls'}
    assert kind in KINDS and action in ALPHAS[1:]
    d=obs['decision'];c=obs['controls']
    if 'losses' not in d:return None
    l=d['losses'];probe=[l['0.0'][g]-l[action][g] for g in ['A','B']]
    controls=[c[action]['margin']-c['0.0']['margin'],c['0.0']['entropy']-c[action]['entropy']]
    x=np.array(probe if kind=='input' else controls if kind=='confidence' else probe+controls,dtype=float)
    assert np.isfinite(x).all();return x
def choose(obs,model):
    values=[0.]
    for action in ALPHAS[1:]:
        x=features(obs,model['kind'],action)
        if x is None:return '0.0',[0.,0.,0.,0.]
        m=model['actions'][action]
        values.append(float(((x-np.array(m['mean']))/np.array(m['scale']))@np.array(m['coef'])+m['intercept']))
    assert np.isfinite(values).all()
    return ALPHAS[int(np.argmax(values))],values
