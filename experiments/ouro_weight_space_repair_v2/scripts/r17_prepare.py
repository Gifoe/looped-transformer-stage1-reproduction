"""Freeze a single paired cue before effects; no accuracy or method candidate."""
import numpy as np
from r17_common import ROOT,LOCK,read,sha,jsonlines,save


def main():
    assert not any((ROOT/p).exists() for p in [LOCK,'results/r17_paired_responses.jsonl','logs/r17_gpu_budget.json'])
    previous=read('configs/r16_NATIVE_KL_PILOT_LOCK.json')
    authority={r['id']:r for s in ['discovery','selection'] for r in jsonlines('results/r7_'+s+'_development.jsonl')}
    ordered=[dict(id=r['id'],split=s) for s in ['discovery','selection'] for r in jsonlines('data/slcd_'+s+'.jsonl')];assert len(ordered)==576
    cohort=[]
    for j in np.random.default_rng(2021).choice(576,size=12,replace=False):
        c=ordered[int(j)].copy();c['candidate']=authority[c['id']]['predictions']['M0_R6'];cohort.append(c)
    ledgerfiles=['logs/r7_gpu_budget.json','logs/r9_gpu_budget.json','logs/r10_gpu_budget.json','logs/r11_gpu_budget.json','logs/r15_gpu_budget.json','logs/r16_gpu_budget.json']
    hashes={p:sha(p) for p in ledgerfiles};total=sum(read(p)['used_seconds'] for p in ledgerfiles);assert abs(total-17742.752264022827)<1e-6
    sources=[p for p in previous['source_sha256'] if not p.startswith('r16_')]+[p.name for p in sorted((ROOT/'scripts').glob('r17_*.py'))]
    spec=dict(status='frozen_observational_probe',new_method_candidates=0,heldout_access_authorized=False,external_benchmarks_authorized=False,cohort=cohort,
        cohort_selection='Uniform12 from original ordered discovery384+selection192, NumPy default_rng2021 choice without replacement. One existing R6-predicted candidate per question; no correctness, margin, R15/R16 outcome or response feature used.',
        source_sha256={p:sha('scripts/'+p) for p in sources},input_sha256=previous['input_sha256'],model_identity=previous['model_identity'],
        intervention=dict(original_cue='Answer:',changed_cue='Response:',real='Replace exactly final current-answer suffix; preserve question,25 demonstrations and delimiters.',reference='Literal N0 Answer: -> literal Response:',
            claim='Fixed actual input intervention, not known pure nuisance; one wording only. No altered-prompt accuracy is evaluated.'),
        measurement=dict(original='gC=.5*(zC4-zC1), computed FP32; original real states cached.',response='vC=center_pQ4(gC_changed.double-gC_original.double); e=center_pQ4(gN_original.double).',
            primary=['R2(e,vN) and signed cosine(e,vN)','R2(vN,vQ) and signed cosine(vN,vQ)'],
            shared_metric='All primary vectors use the same original pQ4 full-vocabulary FP64 Fisher metric. R2=cosine^2 is fraction captured by best signed scalar projection, not a deployed coefficient.',
            controls='One seed2021 full-vocabulary permutation for e or vN, then re-center in pQ4. Also reference-only alignment under original pN4 and static e/gQ alignment as diagnostics.',
            extra='Identity response error ||vQ-vN||²/(||vQ||²+||vN||²); signed least-squares response slope, diagnostic only.',
            zero_rule='Norm<=64eps64*maxabs(centeredvector)^2 means undefined direction; save bool mask with zero placeholder. Do not interpret placeholder as observed cosine0.',
            prefix='All four contexts independently encode original full continuation. Same literal consumed prefix; any unsupported alignment fails full probe rather than filtering.'),
        analysis=dict(units='12 questions equally weighted after within-question position average. Report per-position records and coverage. No token independence or correctness metric.',
            uncertainty='10000 whole-question bootstrap draws seed2021; descriptive95%intervals. Exact4096 paired sign flips for R2 advantage over fixed shuffle, exploratory symmetry test.',
            operational_screen='High equal-question mean prefix projection only if lower95%R2>.5 AND reference vs shuffle p<.05. Conditional on that, low equal-question mean prefix transfer projection supported for this cue if upper95%transferR2<.5. High mean prefix positive transfer only if lower95%transferR2>.5 and lower95%signedcos>0. Otherwise inconclusive.',
            coverage='Any undefined primary/shuffle position blocks operational inferential screen; all raw positions/questions still reported.',
            threshold='R2=.5 is a fixed threshold for equal-question means of per-prefix projection fractions; NOT pooled energy. Diagnostic only, never a decoding gate or tuned parameter.',
            stop='No new decoder, fitted threshold, alternate cue, sample expansion, heldout, external task or automatic model experiment. Findings inform theoretical next-step review only.'),
        budget=dict(original_cap_seconds=21600,inherited_ledger_sha256=hashes,inherited_total_seconds=total,limit_seconds=120,termination_grace_seconds=15,
            original_complete_ARC_worker_seconds_already_used=5625.518387556076,
            scope='One post-evaluation explanatory observation, actual load/worker/failure time all charged; six old ledgers preserved. Same GPU mutex and CPU nice+15, no other processes altered, no automatic retry.'),
        limitations='Does not establish a universal transport theorem or a new successful method. A cue direction can be unrelated to the natural reference loop increment; that would make this probe inconclusive about broader calibration failures.')
    save(LOCK,spec);print('R17_FROZEN',sha(LOCK),len(cohort),total)


if __name__=='__main__':main()
