"""Quantify measured state changes; geometry is descriptive, not semantic evidence."""
import argparse
import csv
import json
import torch
import torch.nn.functional as F
from linux_runtime import ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',default='causal_original_shape');args=ap.parse_args()
    out=ROOT/'fresh'/args.study
    assert (out/'COMPLETE.json').exists()
    rows=[]
    for p in sorted(out.glob('batch_*.json')):
        d=json.loads(p.read_text(encoding='utf8'));bid=int(p.stem.split('_')[1])
        st={c:torch.load(out/f'batch_{bid:03d}_{c}_states.pt',map_location='cpu',weights_only=True) for c in ['link','cut']}
        for x,l,kind in st['link']:
            if kind=='attention':continue
            a,b=st['link'][(x,l,kind)],st['cut'][(x,l,kind)]
            for j in range(a.shape[1]):
                an,bn=a[:,j].norm(dim=-1),b[:,j].norm(dim=-1)
                diff=(a[:,j]-b[:,j]).norm(dim=-1)
                cos=F.cosine_similarity(a[:,j],b[:,j],dim=-1)
                for i,c in enumerate(d['cases']):
                    rows.append(dict(question_id=c['question_id'],receiver=x,selected=c['selected'] and c['target_receiver']==x,
                        position_kind='answer_prefix' if j==0 else 'last_partner_visible',layer=l,tensor=kind,
                        link_l2=float(an[i]),cut_l2=float(bn[i]),delta_l2=float(diff[i]),
                        relative_delta_l2=float(diff[i]/bn[i].clamp_min(1e-12)),cosine=float(cos[i])))
    with (out/'state_dynamics.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(dict(state_measurements=len(rows))))


if __name__=='__main__':main()
