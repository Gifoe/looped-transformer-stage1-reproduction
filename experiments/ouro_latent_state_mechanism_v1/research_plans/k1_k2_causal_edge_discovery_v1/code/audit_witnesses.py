import pathlib,json,torch,datetime,hashlib
R=pathlib.Path('/root/ouro-causal-edge-sol-20261117');p=pathlib.Path('/root/autodl-tmp/ouro-causal-edge-sol-20261117/raw/TECHNICAL_ATTENTION_WITNESSES.pt');raw=torch.load(p,map_location='cpu',weights_only=False);torch.set_num_threads(4);errors=[]
for name,records in raw.items():
 for key,w in records.items():
  q,k,v=[w[n].double() for n in ['query_t','key_causal_prefix','value_causal_prefix']];a,b=w['source_span'];p0=((q@k.transpose(-1,-2))*w['scale']).softmax(-1);base=(p0@v)[:,:,0];kk=k.clone();vv=v.clone()
  if w['type'] in ['K','KV']:kk[:,:,a:b]=w['donor_key_source'].double()
  if w['type'] in ['V','KV']:vv[:,:,a:b]=w['donor_value_source'].double()
  pc=((q@kk.transpose(-1,-2))*w['scale']).softmax(-1);counter=(pc@vv)[:,:,0];err=max(float((base-w['base_reconstructed']).abs().max()),float((counter-w['counter_reconstructed']).abs().max()));assert err<1e-10;after=(w['native_SDPA_row'].double()+counter-base).to(torch.bfloat16);assert torch.equal(after.view(torch.uint8),w['after_BF16'].view(torch.uint8));assert abs(float(p0.sum(-1).sub(1).abs().max()))<1e-12 and pc.shape[-1]==w['target']+1;errors.append(err)
assert len(raw)==9
(R/'CPU_ATTENTION_WITNESS_AUDIT.json').write_text(json.dumps({'status':'PASS','candidates':9,'max_absolute_reconstruction_error':max(errors),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'scope':'CPUFP64 actualpostRoPEQ/K/V rowcounterfactual, fullcausalprefixnormalization and exactBF16writeback; noGPU/model','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2));print('CPU_EDGE_WITNESS_PASS')
