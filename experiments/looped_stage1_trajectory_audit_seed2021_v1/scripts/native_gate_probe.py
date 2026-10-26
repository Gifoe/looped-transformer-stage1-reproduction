"""Released Ouro ponder gate, evaluated as an offline sample-level readout."""
import json
import torch
from ouro_audit import ROOT,prompt_for,save

def gate_cdf(gates):
    remaining=torch.ones_like(gates[0][0,-1,0]);pdf=[]
    for i,gate in enumerate(gates):
        rate=torch.sigmoid(gate[0,-1,0])
        if i<len(gates)-1:
            pdf.append(rate*remaining);remaining=remaining*(1-rate)
        else:pdf.append(remaining)
    pdf=torch.stack(pdf)
    return pdf,pdf.cumsum(0)

def run(tok,m,budget):
    checks=[]
    for text in ['The Solar System contains planets and','Question: What is 2 + 2?\nAnswer:']:
        x=tok(text,return_tensors='pt').to('cuda')
        _,states,gates=m.model(**x,use_cache=False);_,cdf=gate_cdf(gates)
        for threshold in [.25,.5,.75]:
            index=next((i for i,value in enumerate(cdf) if float(value)>=threshold),3)
            native=m(**x,use_cache=False,exit_threshold=threshold,logits_to_keep=1).logits
            extracted=m.lm_head(states[index][:,-1:,:])
            assert torch.equal(native,extracted),(threshold,index)
            checks.append(dict(threshold=threshold,depth=index+1,max_logit_error=0))
    path=ROOT/'trajectories/ouro_native_gate.jsonl'
    records=[]
    for split in ['validation','ouro_test']:
        for line in (ROOT/'data'/(split+'.jsonl')).read_text(encoding='utf-8').splitlines():
            row=json.loads(line);budget.check();prompt,_=prompt_for(row)
            x=tok(prompt,return_tensors='pt').to('cuda')
            _,_,gates=m.model(**x,use_cache=False);pdf,cdf=gate_cdf(gates)
            records.append(dict(id=row['id'],split=split,gate_logits=[float(g[0,-1,0]) for g in gates],exit_pdf=pdf.float().cpu().tolist(),exit_cdf=cdf.float().cpu().tolist(),arithmetic='same BF16 sigmoid/products/cumsum as released forward'))
    path.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    save('results/native_gate_parity.json',dict(passed=True,checks=checks,records=len(records),scope='last prompt token; offline sample-level CDF exit; no actual adaptive wall-clock reduction'))
