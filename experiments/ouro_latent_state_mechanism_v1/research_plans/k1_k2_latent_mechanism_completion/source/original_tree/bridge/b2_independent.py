"""Independent explicit 48-layer bridge wiring replay. No production bridge imports/hooks."""
import sys,json,hashlib,torch
if not __debug__:raise RuntimeError('optimized Python forbidden')
def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def cpu(x):return x.detach().cpu().contiguous().clone()
def versions(m):return tuple((n,id(v),v.data_ptr(),v._version) for n,v in list(m.named_parameters())+list(m.named_buffers()))
class UndefinedReplayControl(RuntimeError):pass
@torch.inference_mode()
def replay(model,q,*,boundary=None,delta=None,sign=-1,norm_preserve=False,x_branch=None,a_branch=None,mlp_delta=None,mlp_sign=-1,mlp_norm_preserve=False):
 assert set(q)=={'id','split','ids','R','targets','input_ids_sha256','prefix_len'} and q['split'] in ('train','validation') and sign in (-1,1)
 assert hashlib.sha256(json.dumps(q['ids']).encode()).hexdigest()==q['input_ids_sha256']
 body=model.model;R=q['R'];assert not model.training and body.total_ut_steps==2 and len(body.layers)==24 and body.config.num_hidden_layers==24 and R==len(q['ids'])-1 and 0<q['prefix_len']<R and not getattr(body,'has_sliding_layers',False)
 assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules());guard=versions(model);ids=torch.tensor([q['ids']],device=next(model.parameters()).device);x=body.embed_tokens(ids);E=cpu(x);cp=torch.arange(R+1,device=x.device);pos=cp.unsqueeze(0);mod=sys.modules[type(body).__module__];mask=mod.create_causal_mask(config=body.config,input_embeds=x,attention_mask=None,cache_position=cp,past_key_values=None,position_ids=pos);rope=body.rotary_emb(x,pos);steps=[];sites={};branch={};nonr={};states=[]
 if delta is not None:assert set(delta)==set(range(9,16))
 if mlp_delta is not None:assert set(mlp_delta)==set(range(9,16)) and mlp_sign in (-1,1)
 for ut in range(2):
  if ut==1:
   B=cpu(x)
   if boundary is not None:
    assert boundary.shape==x.shape and boundary.dtype==x.dtype and torch.isfinite(boundary).all();x=boundary.to(x.device).clone()
  for i,layer in enumerate(body.layers):
   if ut==1 and i==8:Q9=cpu(x)
   assert getattr(layer,'attention_type',None)=='full_attention';kw=dict(attention_mask=mask,position_ids=pos,past_key_value=None,use_cache=False,cache_position=cp,position_embeddings=rope,current_ut=ut)
   if ut==1 and (9<=i<=15 or i==17):
    residual=x;ih=digest(x[:,:R,:]);norm=layer.input_layernorm(x);att,_=layer.self_attn(hidden_states=norm,**kw);a=layer.input_layernorm_2(att)
    if 9<=i<=15:
     before=a[:,R,:].clone();exact=before.double();scale=torch.ones((),dtype=torch.float64,device=x.device)
     if delta is not None:
      d=delta[i];assert d.shape==before.shape and d.dtype==torch.float64 and torch.isfinite(d).all();exact=exact+sign*d.to(x.device)
      if norm_preserve:
       n0=before.double().norm();n1=exact.norm()
       if float(n1)==0 and float(n0)!=0:raise UndefinedReplayControl('zero upstream proposal norm with nonzero receiver norm')
       if float(n1)!=0:scale=n0/n1;exact=exact*scale
     after=exact.to(before.dtype);a=a.clone();a[:,R,:]=after;sites[i]=dict(before=cpu(before),after=cpu(after),exact_before_cast=cpu(exact),normalization_scale_fp64=cpu(scale),cast_residual=cpu(after.double()-exact),effective_delta=cpu(after.double()-before.double()))
    y=residual+a
    if i==17:
     xr=residual[:,R,:];ar=a[:,R,:];xb=xr if x_branch is None else x_branch.to(x.device);ab=ar if a_branch is None else a_branch.to(x.device);assert xb.shape==xr.shape and ab.shape==ar.shape and xb.dtype==xr.dtype and ab.dtype==ar.dtype and torch.isfinite(xb).all() and torch.isfinite(ab).all();y=y.clone();y[:,R,:]=xb+ab
     branch.update(actual_x_R=cpu(xr),actual_a_R=cpu(ar),used_x_R=cpu(xb),used_a_R=cpu(ab),used_y_R=cpu(y[:,R,:]),MLP_skip_R=cpu(y[:,R,:]),non_R_x_sha256=ih,non_R_attention_sha256=digest(a[:,:R,:]))
    skip=y;b=layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(y)))
    if 9<=i<=15:
     mbefore=b[:,R,:].clone();mexact=mbefore.double();mscale=torch.ones((),dtype=torch.float64,device=x.device)
     if mlp_delta is not None:
      md=mlp_delta[i];assert md.shape==mbefore.shape and md.dtype==torch.float64 and torch.isfinite(md).all();mexact=mexact+mlp_sign*md.to(x.device)
      if mlp_norm_preserve:
       n0=mbefore.double().norm();n1=mexact.norm()
       if float(n1)==0 and float(n0)!=0:raise UndefinedReplayControl('zero MLP proposal norm with nonzero receiver norm')
       if float(n1)!=0:mscale=n0/n1;mexact=mexact*mscale
     mafter=mexact.to(mbefore.dtype);b=b.clone();b[:,R,:]=mafter
     for k,v in dict(before=mbefore,after=mafter,exact_before_cast=mexact,normalization_scale_fp64=mscale,cast_residual=mafter.double()-mexact,effective_delta=mafter.double()-mbefore.double()).items():sites[i]['mlp_'+k]=cpu(v)
    x=skip+b
    if i==17:branch.update(b_R=cpu(b[:,R,:]),postMLP_R=cpu(x[:,R,:]),non_R_after_mlp_sha256=digest(x[:,:R,:]))
    else:nonr[i]=dict(input_nonR_sha256=ih,output_nonR_sha256=digest(x[:,:R,:]))
   else:x=layer(x,**kw)
   assert torch.is_tensor(x) and torch.isfinite(x).all();steps.append((ut,i))
  x=body.norm(x);states.append(x.clone())
  if ut==1:branch.update(finalnorm_R=cpu(x[:,R,:]),non_R_final_norm_sha256=digest(x[:,:R,:]))
 z=model.lm_head(states[1])[:,R,:];assert z.shape==(1,49152) and z.dtype==torch.bfloat16 and torch.isfinite(z).all() and versions(model)==guard and steps==[(u,i) for u in (0,1) for i in range(24)]
 return dict(logits=cpu(z),E=E,B=B,Q9_actual=Q9,sites=sites,branch=branch,nonr=nonr,steps=steps,runtime_torch=torch.__version__,scope='Independent explicit layer wiring replay; shared pinned decoder/mask/checkpoint, no production bridge code or hooks')
