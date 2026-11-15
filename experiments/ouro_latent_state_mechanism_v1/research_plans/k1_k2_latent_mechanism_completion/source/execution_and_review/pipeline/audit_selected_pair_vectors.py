"""Supplemental saved-vector audit of six original fixed pairs; no primary changes."""
import argparse,pathlib,json,hashlib,sys,torch
PAIRS=[('Q_R_E','H10','H00'),('Q_R_B','H11','H01'),('R_Q_E','H01','H00'),('R_Q_B','H11','H10'),('Q9_restoration','U_Q9_N','H01'),('Q9_transfer_loss','H11','N_Q9_U')]
SITES=('L1_R','L9_R','L10_a_R','L18_x_R','L18_a_R','L18_y_R','finalnorm_R')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--stage',required=True,choices=['train64','train1030','validation299']);a=ap.parse_args();b=pathlib.Path(a.root);sys.path[:0]=[str(b/'storage'),str(b/'code')];from ab1_prefix_codec import load
 d=b/'data'/a.stage;o=b/'analysis_results'/a.stage;pp=d/'PENDING.json';apath=d/'INDEPENDENT_COVERAGE_AUDIT.json';rp=o/'rows.jsonl';p=json.loads(pp.read_text());audit=json.loads(apath.read_text());assert p['status']=='COMPLETE' and audit['status']=='PASS' and audit['pending_sha256']==sha(pp);analysis={r['id']:r for line in rp.read_text().splitlines() if (r:=json.loads(line))};before={str(x.relative_to(b)):sha(x) for x in (pp,apath,rp,pathlib.Path(__file__))};records=[];earliest=[]
 for c in p['cases']:
  mp=d/c['metadata_file'];assert sha(mp)==c['metadata_sha256'];m=json.loads(mp.read_text());t=load(m,d);r=analysis[c['id']]
  for edge,left,right in PAIRS:
   changed=[];observed=[]
   for site in SITES:
    lk=left+'_'+site;rk=right+'_'+site
    if lk not in t or rk not in t:continue
    x=t[lk];y=t[rk];assert x.shape==y.shape and x.dtype==y.dtype;equal=same(x,y);xd=x.reshape(-1).double();yd=y.reshape(-1).double();nx=xd.norm();ny=yd.norm();dose=(xd-yd).norm()
    if edge in ('Q_R_E','Q_R_B') and site=='L1_R':assert equal,'Q-only boundary changed R input'
    if edge.startswith('Q9_') and site in ('L1_R','L9_R'):assert equal,'Q9 transplant changed protected R site'
    observed.append(site)
    if not equal:changed.append(site)
    records.append(dict(id=c['id'],edge=edge,left=left,right=right,site=site,bitwise_equal=equal,l2_displacement=float(dose),left_norm=float(nx),right_norm=float(ny),cosine=float(xd.dot(yd)/(nx*ny)) if nx and ny else None,relative_dose_to_right=float(dose/ny) if ny else None,candidate_count=len(m['targets']),natural_group=r['natural_group'],natural_no_tie=r['natural_no_tie']))
   required=['L1_R','L9_R'] if edge.startswith('Q9_') else ['L1_R'];assert set(required)<=set(observed)
   earliest.append(dict(id=c['id'],edge=edge,observed_sites=observed,earliest_observed_selected_site=changed[0] if changed else None,interpretation='Earliest changed among saved selected sites; not first physical writer. Later nonzero change is not required.'))
 assert all(sha(b/k)==v for k,v in before.items());out=o/'SELECTED_PAIR_VECTORS.json';assert not out.exists();out.write_text(json.dumps(dict(status='PASS',stage=a.stage,source_sha256=before,records=records,earliest=earliest,scope='Independent saved-tensor check of original four boundary and two Q9 pairs. No primary metric or head selection changed.'),indent=2)+'\n');print('PASS',len(records))
if __name__=='__main__':main()
