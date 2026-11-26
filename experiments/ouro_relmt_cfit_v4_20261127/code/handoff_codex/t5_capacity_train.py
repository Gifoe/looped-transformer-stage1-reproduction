"""The frozen T5 training step adapted for capacity-matched controller families."""
import bisect
import random
import time

import torch
from torch.nn import functional as F

from v4_t5_locked import DEPTHS, RECIPE, STEPS, cloze_option_logits_differentiable, forward_with_hook_all_tokens, make_candidate_row


def train(archive, model, tokenizer, bundle, arm, seed, trust, text, checkpoint, lock_sha, family, initialize):
    torch.manual_seed(seed)
    controller = initialize(family).to(model.device).requires_grad_(True)
    optimizer = torch.optim.AdamW(controller.parameters(),lr=RECIPE["learning_rate"],weight_decay=RECIPE["weight_decay"])
    matched = arm in ("T5_Poison_MedOnly_Raw","T5_Poison_CFIT_Rescued")
    stream = "medmcqa_matched_poison" if matched else arm
    source_rng = random.Random(f"t5/{stream}/{seed}/source")
    item_rngs = [random.Random(f"t5/{stream}/{seed}/{source}") for source,_,_ in bundle]
    groups, cumulative = [], []
    for _,weight,items in bundle:
        item_rows, item_weights = zip(*items)
        item_cum = []
        for value in item_weights:
            item_cum.append((item_cum[-1] if item_cum else 0.) + value)
        groups.append((item_rows,item_cum))
        cumulative.append((cumulative[-1] if cumulative else 0.)+weight)
    start, elapsed, losses, n_forwards = 0, 0., [], 0
    if checkpoint.exists():
        pre = torch.load(checkpoint,map_location="cpu",weights_only=False)
        if pre["lock_sha256"]!=lock_sha:
            raise ValueError("Refuse to resume a T5 job from different inputs")
        controller.load_state_dict(pre["state"])
        optimizer.load_state_dict(pre["optim"])
        source_rng.setstate(pre["source_rng"])
        for rng,state in zip(item_rngs,pre["item_rngs"],strict=True):
            rng.setstate(state)
        start,elapsed,losses,n_forwards = pre["step"],pre["seconds"],pre["losses"],pre["score_microbatches"]
    teacher_cache = {}

    def penalty(seq, depth):
        ids = torch.tensor(seq,device=model.device,dtype=torch.long).unsqueeze(0)
        mask = torch.ones_like(ids)
        target = ids[:,1:].flatten()
        key = depth,tuple(seq)
        if key not in teacher_cache:
            with torch.no_grad():
                hidden = forward_with_hook_all_tokens(model,ids,mask,depth,net=None)
                logits = model.lm_head(hidden[:,:-1]).float().squeeze(0)
                teacher_cache[key] = (float(F.cross_entropy(logits,target)),
                                      logits.softmax(-1)[:,tokenizer.eos_token_id].cpu())
        hidden = forward_with_hook_all_tokens(model,ids,mask,depth,net=controller)
        logits = model.lm_head(hidden[:,:-1]).float().squeeze(0)
        default, native_eos = teacher_cache[key]
        margin = .50 if arm=="T5_CFIT_Top6_Trust_4096" else .55
        change = (logits.softmax(-1)[:,tokenizer.eos_token_id]-native_eos.to(logits.device)).abs().mean()
        return F.relu(F.cross_entropy(logits,target)-default-margin) + change

    begin=time.perf_counter()
    for step in range(start+1,STEPS+1):
        chosen = []
        for _ in range(RECIPE["effective_batch"]):
            pos=bisect.bisect_right(cumulative,source_rng.random()*cumulative[-1])
            items, weights=groups[pos]
            selected=bisect.bisect_right(weights,item_rngs[pos].random()*weights[-1])
            chosen.append(items[selected])
        length=max(len(it["question"])+max(map(len,it["options"])) for it in chosen)
        options=max(len(it["options"]) for it in chosen)
        size=1 if length>1400 or length*options>5200 else (2 if length>650 or length*options>2600 else len(chosen))
        depth=DEPTHS[(step-1)%len(DEPTHS)]
        optimizer.zero_grad(set_to_none=True)
        total=0.
        with torch.enable_grad():
            for start_row in range(0,len(chosen),size):
                subset=chosen[start_row:start_row+size]
                crows=[make_candidate_row(it["id"],it["question"],it["options"],it["gold"],dataset=it["dataset"]) for it in subset]
                logits,_,_=cloze_option_logits_differentiable(archive,model,tokenizer,crows,depth,controller)
                loss=sum(F.cross_entropy(x.unsqueeze(0),torch.tensor([it["gold"]],device=x.device))
                         for x,it in zip(logits,subset,strict=True))/len(chosen)
                loss.backward()
                n_forwards+=1
                total+=float(loss.detach())
            if trust and step%3==0:
                loss=trust*penalty(text[step%len(text)],depth)
                loss.backward()
                total+=float(loss.detach())
        grad=torch.nn.utils.clip_grad_norm_(controller.parameters(),RECIPE["clip"])
        if not grad.isfinite().item():
            raise FloatingPointError("A registered Rank-1 arm has non-finite gradients")
        optimizer.step()
        if family == "ortho_rank2":
            controller.retract()
        losses=(losses+[total])[-70:]
        if step%70==0 or step==STEPS:
            staging=checkpoint.with_suffix(".partial")
            torch.save(dict(state=controller.state_dict(),optim=optimizer.state_dict(),source_rng=source_rng.getstate(),
                            item_rngs=[rng.getstate() for rng in item_rngs],step=step,seconds=elapsed+time.perf_counter()-begin,
                            losses=losses,score_microbatches=n_forwards,lock_sha256=lock_sha),staging)
            staging.replace(checkpoint)
            print("T5_TRAIN",arm,seed,step,STEPS,round(elapsed+time.perf_counter()-begin,1),flush=True)
    return controller,dict(seconds=elapsed+time.perf_counter()-begin,steps=STEPS,n_params=sum(p.numel() for p in controller.parameters()),
                           score_microbatches=n_forwards,loss_final_mean=sum(losses)/len(losses))
