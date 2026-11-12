import numpy as np
from e2_state_intervention import raw_probe_span,project
rng=np.random.default_rng(12);w=rng.normal(size=(9,5));std=rng.uniform(.4,2,size=8);G,s,t=raw_probe_span(w,std);O,os,ot=raw_probe_span(w[:,[0,1,2,4]],std);d=rng.normal(size=8);assert np.linalg.norm(O-G@(G.T@O))<1e-12;assert np.linalg.norm(project(d,G)-project(d,O))**2 >= -1e-12;assert G.shape[1]==4 and O.shape[1]==3;print('PASS offered target span subset/ranks; no gold selected')
