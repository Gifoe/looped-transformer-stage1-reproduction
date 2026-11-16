import pathlib,json,torch,hashlib
R=pathlib.Path('/root/ouro-computation-sol-20261116');qs={x['id']:x for x in map(json.loads,(R/'B_INPUTS.jsonl').read_text().splitlines())};torch.set_num_threads(4)
def check(z,r,targets):
 assert z.shape==(1,49152) and z.dtype==torch.bfloat16
 assert hashlib.sha256(z.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==r['logits_sha256'];v=z.reshape(-1)[targets].float().tolist();assert v==r['candidate_logits'];ties=[i for i,x in enumerate(v) if x==max(v)];assert ties==r['ties'] and ties[0]==r['prediction'];lp=z.double().log_softmax(-1).reshape(-1)[targets].tolist();assert max(abs(a-b) for a,b in zip(lp,r['scores']))<1e-5
