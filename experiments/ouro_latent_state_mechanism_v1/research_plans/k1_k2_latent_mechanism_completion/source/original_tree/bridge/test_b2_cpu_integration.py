"""CPU schema/math integration fixture. No GPU; not scientific or model gate."""
import pathlib,tempfile,json,hashlib,torch
from b2_audit import site_math,branch_control,score,checked
from b2_analyze import contrasts,gap

def main():
 with tempfile.TemporaryDirectory() as td:
  root=pathlib.Path(td);stage=root/'b2_data/treatments/train64';stage.mkdir(parents=True);meta=dict(id='fake',split='train',R=2,prefix_len=1,targets=[330,340],AB_metadata_path='fake_ab.json',AB_metadata_sha256='source-bound-fixture',reference_metadata_path='fake_reference.json',reference_metadata_sha256='source-bound-fixture',producer_identity={'backend':'CPU_FIXTURE'},source_identity={'origin':'AB_FIXTURE'},source_freeze_sha256='freeze-fixture');mp=stage/'fake.json';mp.write_text(json.dumps(meta));h=hashlib.sha256(mp.read_bytes()).hexdigest();pending=dict(status='COMPLETE',kind='treatments',stage='train64',split='train',n=1,cases=[dict(id='fake',metadata_file='fake.json',metadata_sha256=h,tensor_raw_sha256='codec-fixture')],producer_identity=meta['producer_identity'],source_freeze_sha256=meta['source_freeze_sha256']);pp=stage/'PENDING.json';pp.write_text(json.dumps(pending));_,restored=checked(root,'b2_data/treatments/train64/fake.json',h);assert restored==meta
  audit=dict(status='PASS',pending_sha256=hashlib.sha256(pp.read_bytes()).hexdigest());(stage/'INDEPENDENT_COVERAGE_AUDIT.json').write_text(json.dumps(audit));assert json.loads((stage/'INDEPENDENT_COVERAGE_AUDIT.json').read_text())['pending_sha256']==hashlib.sha256(pp.read_bytes()).hexdigest()
  trace={9:{'input_nonR_sha256':'a','output_nonR_sha256':'b'}};disk=json.loads(json.dumps(trace));assert {str(k):v for k,v in trace.items()}=={str(k):v for k,v in disk.items()}
  before=torch.tensor([[3.,4.]],dtype=torch.bfloat16);delta=torch.tensor([[1.,-2.]],dtype=torch.float64);exact=before.double()-delta;scale=before.double().norm()/exact.norm();exact*=scale;after=exact.to(before.dtype);t={};prefix='M_site_09_mlp_'
  for key,value in dict(before=before,after=after,exact_before_cast=exact,normalization_scale_fp64=scale,cast_residual=after.double()-exact,effective_delta=after.double()-before.double()).items():t[prefix+key]=value
  site_math(t,'M',9,delta,-1,True,True)
  shifted=torch.nextafter(scale,torch.tensor(float('inf'),dtype=torch.float64));shift_exact=(before.double()-delta)*shifted;shift_after=shift_exact.to(before.dtype);shifted_t=dict(t)
  for k,v in dict(normalization_scale_fp64=shifted,exact_before_cast=shift_exact,after=shift_after,cast_residual=shift_after.double()-shift_exact,effective_delta=shift_after.double()-before.double()).items():shifted_t[prefix+k]=v
  site_math(shifted_t,'M',9,delta,-1,True,True)
  tampered=dict(shifted_t);tampered[prefix+'normalization_scale_fp64']=shifted+1e-3
  try:site_math(tampered,'M',9,delta,-1,True,True)
  except AssertionError:pass
  else:raise AssertionError('invalid scale accepted')
  bad=dict(t);bad[prefix+'after']=after+1
  try:site_math(bad,'M',9,delta,-1,True,True)
  except AssertionError:pass
  else:raise AssertionError('corrupt cast accepted')
  r=branch_control(before,torch.tensor([[6.,8.]],dtype=torch.bfloat16),delta,'norm',1);assert torch.equal(r['control_after'],torch.tensor([[6.,8.]],dtype=torch.bfloat16))
  z=torch.zeros(49152,dtype=torch.bfloat16);z[330]=z[340]=2;v=z[[330,340]];lse=float(z.double().logsumexp(0));s=dict(candidate_logits=[2.,2.],ties=[0,1],prediction=0,logsumexp=lse,scores=[2-lse,2-lse]);score(z,s,[330,340]);assert {k:len(v) for k,v in contrasts().items()}=={'attention':6,'L18_branch':12,'MLP_secondary':6};assert gap([3.,1.],0,1)['meanwrong']==2
 print('CPU_B2_SCHEMA_MATH_INTEGRATION_PASS')
if __name__=='__main__':main()
