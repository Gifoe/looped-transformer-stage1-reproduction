"""Official answer-letter scoring with raw logits and auditable causal positions."""
import torch
import games as G
import session as S
from link import SIDES, DEVICE, partner


def prepare(pair, items, style='pair'):
    episodes = [G.qa_episode(it, style) for it in items]
    built = [S.build(ep, pair.toks) for ep in episodes]
    batch = S.collate(built, {x: pair.toks[x].pad_token_id for x in SIDES})
    positions = {}
    for x in SIDES:
        prefix = S.ids(pair.toks[x], 'The answer is')
        positions[x] = torch.tensor([[e[x]['labels'][len(prefix)][0]] for e in built], device=DEVICE)
        assert all([t for _,t,_ in e[x]['labels'][:len(prefix)]] == prefix for e in built)
    if style == 'pair':
        for x in SIDES:
            for i, e in enumerate(built):
                y = partner(x)
                assert e[y]['time'][int(positions[y][i,0]) + 1] > e[x]['time'][int(positions[x][i,0])]
    return batch, positions, built


def letter_logits(pair, last, positions):
    logits = {}
    for x in SIDES:
        lid = [S.ids(pair.toks[x], ' ' + letter)[0] for letter in G.LETTERS]
        n = last[x].shape[1]
        idx = torch.arange(last[x].shape[0], device=DEVICE) * n + positions[x][:,0]
        logits[x] = pair.token_logits(x, last[x], idx)[:,lid].detach().float()
    return logits


def outcomes(logits, items):
    out = {}
    for x, lg in logits.items():
        prob = lg.softmax(-1).cpu()
        lg = lg.cpu()
        rows = []
        for i,it in enumerate(items):
            gold = it['answer']; pred = int(lg[i].argmax())
            wrong = torch.cat([lg[i,:gold], lg[i,gold+1:]])
            rows.append(dict(prediction=pred, gold=gold, correct=pred==gold,
                letter_logits=lg[i].tolist(), letter_probabilities=prob[i].tolist(),
                correct_answer_margin=float(lg[i,gold]-wrong.max())))
        out[x] = rows
    return out


def frozen_readout_rows(pair):
    """Exact stored head weights for an offline high-precision dot-product diagnostic."""
    from link import inner
    out={}
    for x in SIDES:
        lid=[S.ids(pair.toks[x],' '+letter)[0] for letter in G.LETTERS]
        head=inner(pair.models[x])[1]
        out[x]=dict(weight=head.weight[lid].detach().to('cpu',torch.float64),
            bias=head.bias[lid].detach().to('cpu',torch.float64) if head.bias is not None else None)
    return out


def add_dot64_readout(records,last,positions,readouts):
    """Recompute only the output dot products in FP64; upstream states stay unchanged."""
    for x in SIDES:
        idx=torch.arange(last[x].shape[0],device=last[x].device)
        h=last[x][idx,positions[x][:,0]].detach().to('cpu',torch.float64)
        lg=h@readouts[x]['weight'].T
        if readouts[x]['bias'] is not None:lg=lg+readouts[x]['bias']
        for i,r in enumerate(records[x]):
            g=r['gold'];pred=int(lg[i].argmax())
            r.update(dot64_logits=lg[i].tolist(),dot64_prediction=pred,dot64_correct=pred==g,
                dot64_margin=float(lg[i,g]-torch.cat([lg[i,:g],lg[i,g+1:]]).max()))
