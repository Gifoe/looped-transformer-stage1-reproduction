"""Frozen contract: existing captured R vectors only; descriptive geometry, no site selection."""
import argparse,pathlib,json,sys,hashlib,torch,numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'storage'))
from ab1_prefix_codec import load

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--stage',required=True);a=ap.parse_args();b=pathlib.Path(a.root);d=b/'data'/a.stage;o=b/'analysis_results'/a.stage;p=json.loads((d/'PENDING.json').read_text());heads=np.load(b/'inputs/selected_models.npz');axis_rows=[];assert json.loads((d/'INDEPENDENT_COVERAGE_AUDIT.json').read_text())['status']=='PASS';rows=[]
 for c in p['cases']:
  mp=d/c['metadata_file'];assert hashlib.sha256(mp.read_bytes()).hexdigest()==c['metadata_sha256'];m=json.loads(mp.read_text());t=load(m,d);e=t['source_E'][0,m['R']].double();en=e.norm();base='H11'
  for site,left,right in [('l18_input',t['source_K1_L18_x_R'],t['H11_L18_x_R']),('post_norm',t['source_B'][:,m['R'],:],t['H11_finalnorm_R'])]:
   delta=(right.double()-left.double()).reshape(-1).numpy()
   for control in ('real','permuted'):
    for loop in (1,2):
     key=f'{site}_raw_trainK{loop}_{control}';w=heads[key+'_w'][:-1]/heads[key+'_std'][:,None]
     for family,cols in [('global5',list(range(5))),('offered',[list((330,340,389,414,422)).index(v) for v in m['targets']])]:
      wc=w[:,cols];centered=wc-wc.mean(axis=1,keepdims=True);u,singular,_=np.linalg.svd(centered,full_matrices=False);tol=max(centered.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0);rank=int((singular>tol).sum());assert rank<=len(cols)-1;basis=u[:,:rank];projected=basis@(basis.T@delta);dn=np.linalg.norm(delta)
      axis_rows.append(dict(id=m['id'],site=site,head=key,family=family,rank=rank,rank_cap=len(cols)-1,delta_norm=float(dn),answer_label_axis_projected_norm=float(np.linalg.norm(projected)),answer_label_axis_fraction_squared=float(np.dot(projected,projected)/np.dot(delta,delta)) if dn else None,orthogonal_norm=float(np.linalg.norm(delta-projected))))
  for name,info in m['conditions'].items():
   if info['status']!='OK':continue
   receiver='H00' if name.startswith(('Q_R_E','R_Q_E')) else 'H11' if name.startswith(('Q_R_B','R_Q_B')) or name=='N_Q9_U' else 'H01' if name=='U_Q9_N' else base
   for suffix in ('L1_R','L9_R','L10_a_R','L18_x_R','L18_a_R','L18_y_R','finalnorm_R'):
    key=name+'_'+suffix
    if key not in t:continue
    x=t[key].reshape(-1).double();norm=x.norm();projection=float(x.dot(e)/en) if en else None;reference=t.get(receiver+'_'+suffix);delta=None if reference is None else x-reference.reshape(-1).double()
    rows.append(dict(id=m['id'],condition=name,site=suffix,receiver=receiver,norm=float(norm),embedding_alignment_projection=projection,embedding_alignment_cosine=float(x.dot(e)/(en*norm)) if en and norm else None,embedding_orthogonal_norm=float((x-e*(x.dot(e)/(en*en))).norm()) if en else None,receiver_displacement=float(delta.norm()) if delta is not None else None,receiver_relative_displacement=float(delta.norm()/reference.double().norm()) if delta is not None and reference.double().norm() else None))
 (o/'A_ANSWER_LABEL_GEOMETRY.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in axis_rows))
 (o/'A_VECTOR_GEOMETRY.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(o/'A_VECTOR_SCOPE.json').write_text(json.dumps(dict(status='COMPLETE',n=len(rows),scope='Only existing captured vectors. L10 missing when uncaptured, not imputed. Embedding alignment geometry is descriptive, no probe refit or native-use claim.',missing_L10_capture=not any(r['site']=='L10_a_R' for r in rows)),indent=2)+'\n')
if __name__=='__main__':main()
