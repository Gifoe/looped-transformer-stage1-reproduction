"""Orchestrate matched A/B/C/D fits, development-only selection, and the full panel."""

import argparse
import json
from pathlib import Path
import shutil
from statistics import median
import subprocess

import minimal
from select_external import assess, improved, ordering


SEEDS = (261124137, 261124247, 261124359)
ARMS = ("native", "t1", "t2", "retention", "mixed")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def job(label, script, options):
    log = Path("logs") / f"{label}.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("a", encoding="utf-8") as out:
        print("GPU", label, flush=True)
        subprocess.run(["bash", "gpu_job.sh", label, script, *map(str, options)], stdout=out,
                       stderr=subprocess.STDOUT, check=True)


def weight(recipe, seed, epoch):
    return Path(f"results/fits/{recipe}/{seed}/epoch_{epoch:02d}/controller.pt")


def fit(a, recipe, seed, through, rank):
    target = weight(recipe,seed,through)
    if target.exists() and read(target.with_name("TRAIN.json"))["checkpoint_sha256"] == minimal.sha(target):
        return
    opts = [*a.common, "--data", "data", "--out", target.parent.parent, "--recipe", recipe, "--seed", seed,
            "--through-epoch",through]
    if rank is not None:
        opts += ["--rank-weight", rank]
    job(f"fit_{recipe}_{seed}_e{through}", "train_external.py", opts)


def core(a, arm, chk, phase, depths, freeze=None):
    root=Path(f"results/{phase}/core")
    choices=[*a.common, "--data", "data/panel", "--out", root, "--phase",phase,"--arm",arm,"--depths",*depths]
    if chk:
        choices += ["--checkpoint", chk]
    if freeze:
        choices += ["--freeze",freeze]
    job(f"{phase}_{arm}_core", "evaluate_core.py", choices)
    return root


def develop(a, recipe, seed, epoch, phase):
    arm = f"{recipe}_s{seed}_e{epoch}"
    core_dir = core(a,arm,weight(recipe,seed,epoch),phase,(2,3,4))
    result = assess(core_dir,arm,phase)
    result.update(recipe=recipe, seed=seed, epoch=epoch, weight=str(weight(recipe,seed,epoch)))
    return result


def first_seed(a, recipe, rank):
    seed=SEEDS[0]
    fit(a,recipe,seed,3,rank)
    scores=[develop(a,recipe,seed,epoch,"dev-small") for epoch in (1,2,3)]
    eligible=improved(scores[0],scores[1]) and improved(scores[1],scores[2])
    for epoch in (4,5):
        if not eligible:
            break
        fit(a,recipe,seed,epoch,rank)
        candidate=develop(a,recipe,seed,epoch,"dev-small")
        eligible=improved(scores[-1],candidate)
        scores.append(candidate)
    return scores


def matched_work(a, scores, rank):
    """Confirm a favorable D result if it consumed materially more training work."""
    cs, ds = (scores[r][2] for r in 'CD')
    c, d = (read(Path(row['weight']).with_name('TRAIN.json'))['padded_token_loops'] for row in (cs,ds))
    comparison=dict(matched_epoch=3,C_padded_token_loops=c,D_padded_token_loops=d,relative_excess=d/c-1)
    out=Path('results/compute_matched_D')
    if ordering(ds)>ordering(cs) and d>=c*(1+read(a.study)['paired_compute_trigger']):
        receipt=out/'WORK_STOP.json'
        if not receipt.exists():
            job('fit_D_matched_C_work','train_external.py',
                [*a.common,'--data','data','--out',out,'--recipe','D','--seed',SEEDS[0],
                 '--rank-weight',rank,'--through-epoch',3,'--max-work',c])
        capped=read(receipt)
        run=core(a,'D_compute_matched',Path(capped['checkpoint']),'dev-small',(2,3,4))
        comparison.update(work_receipt=capped,matched_development=assess(run,'D_compute_matched','dev-small'),executed=True)
    else:
        comparison.update(executed=False,why='D did not lead at equal question exposure or consumed less than 10% additional padded token-loop work')
    minimal.dump(Path('results/CD_COMPUTE_AUDIT.json'),comparison)


def freeze(a, selected, refs, development):
    path=Path("results/FREEZE.json")
    win=Path(selected["weight"])
    final=Path("results/FINAL_CONTROLLER.pt")
    if not final.exists():
        tmp=final.with_suffix(".part")
        shutil.copyfile(win,tmp)
        tmp.replace(final)
    if minimal.sha(final) != minimal.sha(win):
        raise ValueError("The deployed controller differs from the development winner")
    slots={arm:dict(path=str(p.resolve()) if p else None, sha256=minimal.sha(p) if p else "native")
           for arm,p in {**refs,"joint":final}.items()}
    stage_code=a.feedback/"stage2/code/stage2_score.py"
    followup=a.feedback/"generation_followup"
    fixture=a.feedback/"stage2/inputs"
    names=("arc_challenge","arc_easy","sciq","mmlu","hellaswag","winogrande","piqa")
    frozen=dict(selection=dict(arm=selected['arm'], recipe=selected['recipe'], epoch=selected['epoch'], seed=selected['seed']),
                selection_sha256=minimal.sha(development), parameters=4096, model_sha256=minimal.sha(a.model/"model.safetensors"),
                panel_sha256=minimal.sha(Path("data/panel/PANEL.json")), arms=slots,
                core_scorer_sha256=minimal.sha(Path("evaluate_core.py")), large_benchmark_scorer_sha256=minimal.sha(Path("evaluate_benchmarks.py")),
                stage2_sha256=minimal.sha(stage_code), mmlu_pro_sha256=minimal.sha(followup/"data/mmlu_pro.parquet"),
                mmlu_pro_code_sha256=minimal.sha(followup/"generation_followup_run.py"),
                seven_requests_sha256={name:minimal.sha(fixture/f"{name}.jsonl") for name in names},
                study_config_sha256=minimal.sha(a.study),primary=read(a.study)["primary"])
    if path.exists() and read(path) != frozen:
        raise ValueError("Selection or formal test inputs changed after locking")
    minimal.dump(path,frozen)
    return path


def final_panel(a, registry):
    plan=read(registry)
    benchmark=[*a.common, "--minimal",a.feedback,"--stage2",a.feedback/"stage2","--unified",a.unified,
               "--followup", a.feedback/"generation_followup", "--historical",a.historical,
               "--out","results/test/benchmark", "--freeze",registry]
    for arm in ("joint", *ARMS):
        path=plan['arms'][arm]['path']
        weight_path=Path(path) if path else None
        core(a,arm,weight_path,"test",(4,),registry)
        job(f"test_{arm}_seven","evaluate_benchmarks.py",[*benchmark,"--arm",arm,"--task","seven"])
        job(f"test_{arm}_mmlu_pro","evaluate_benchmarks.py",[*benchmark,"--arm",arm,"--task","mmlu_pro"])
    for arm in ("joint", *ARMS):
        path=plan['arms'][arm]['path']
        core(a,arm,Path(path) if path else None,"transfer",(2,3,5,6),registry)
    subprocess.run([a.python,"aggregate_external.py","--freeze",str(registry)],check=True)
    subprocess.run([a.python,"report_external.py"],check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("archive","previous","model","feedback","unified","controllability","advantage","generation"):
        p.add_argument(f"--{key}",type=Path,required=True)
    p.add_argument('--historical',type=Path,default=Path('/root/ouro-rank1-functional-theory-generation-20261123'))
    p.add_argument('--study',type=Path,default=Path('configs/study.json'))
    p.add_argument("--development-only",action="store_true")
    p.add_argument("--python",required=True)
    a=p.parse_args()
    from train_external import DEPTHS, QUOTAS_PER_STEP
    study=read(a.study)
    if tuple(study['seeds'])!=SEEDS or tuple(study['train_depths'])!=DEPTHS or tuple(study['batch_source_counts'])!=QUOTAS_PER_STEP:
        raise ValueError('Prespecified schedule does not match the trainer')
    a.common=("--archive",a.archive,"--previous",a.previous,"--model",a.model)
    refpack=a.controllability/"results/weights"
    refs={"native":None,"t1":refpack/"rank1_t1.pt","t2":refpack/"rank1_t2.pt",
          "retention":a.generation/"results/FINAL_CONTROLLER.pt","mixed":a.advantage/"results/fits/mixed/controller.pt"}
    prior_selection=Path("results/DEVELOPMENT_SELECTION.json")
    frozen=Path("results/FREEZE.json")
    if frozen.exists():
        if not a.development_only:
            final_panel(a,frozen)
        return
    for arm, pth in refs.items():
        if arm != "t2":
            core(a,arm,pth,"dev-small",(2,3,4))
    scores={r:first_seed(a,r,None) for r in "AB"}
    first={recipe:max(rows,key=ordering) for recipe,rows in scores.items()}
    chosen=max(first,key=lambda recipe: ordering(first[recipe]))
    rank=0. if chosen=='A' else .25
    for r in "CD":
        scores[r]=first_seed(a,r,rank)
        first[r]=max(scores[r],key=ordering)
    matched_work(a,scores,rank)
    top=sorted(first,key=lambda recipe: ordering(first[recipe]),reverse=True)[:2]
    for arm,pth in refs.items():
        if arm != "t2":
            core(a,arm,pth,"dev-large",(2,3,4))
    large=[]
    for recipe in top:
        final_epoch=first[recipe]['epoch']
        for seed in SEEDS:
            fit(a,recipe,seed,final_epoch,rank if recipe in "CD" else None)
            large.append(develop(a,recipe,seed,final_epoch,"dev-large"))
    recipe_order = {r: tuple(median(column) for column in zip(*(ordering(row) for row in large if row['recipe']==r))) for r in top}
    retained=max(top,key=recipe_order.__getitem__)
    selected=max((row for row in large if row['recipe']==retained),key=ordering)
    minimal.dump(prior_selection,dict(selected=selected, screen=scores, objective_winner=chosen, finalists=top,
                                     seed_results=large, seed_median_order=recipe_order,
                                     priority="significant Native damage count, worst Native and strong-reference gaps, class-balanced means",
                                     early_stop="extend beyond epoch 3 only with consecutive stable development improvements of 0.5 pp, cap 5"))
    mechanism=Path("results/mechanism/SUMMARY.json")
    c,d=(weight(r,SEEDS[0],3) for r in "CD")
    if not mechanism.exists():
        job("mechanism_C_D","mechanism_external.py",[*a.common,"--data","data","--short",c,"--explained",d,"--out",mechanism.parent])
    if read(mechanism)["checkpoint_sha256"] != {"C":minimal.sha(c),"D":minimal.sha(d)}:
        raise ValueError("Mechanism replay does not use the matched epoch-3 pair")
    frozen=freeze(a,selected,refs,prior_selection)
    if not a.development_only:
        final_panel(a,frozen)
    print("DONE",flush=True)


if __name__=="__main__":
    main()
