"""Build the completed research report from audited measured evidence only."""
import json
from pathlib import Path
import pandas as pd
from common import ROOT,BASE,CFG,save

def table(frame,columns):
    rows=[]
    for _,row in frame.iterrows():
        cells=[]
        for key,label,kind in columns:
            value=row[key]
            cells.append('—' if pd.isna(value) else f'{100*value:.2f}' if kind=='percent' else f'{value:.3f}' if kind=='float' else str(value))
        rows.append('| '+' | '.join(cells)+' |')
    return '\n'.join(['| '+' | '.join(label for _,label,_ in columns)+' |','| '+' | '.join('---' for _ in columns)+' |',*rows])

def main():
    audit=json.loads((ROOT/'results/completion_audit.json').read_text());assert audit['passed']
    decision_path=ROOT/'configs/research_decision.json'
    assert decision_path.exists(),'Inspect all measured evidence and write an evidence-grounded research decision before finalizing'
    decision=json.loads(decision_path.read_text(encoding='utf-8'));assert decision['verdict'] and decision['reasoning'] and decision['evidence']
    assert all(decision.get('question_answers',{}).get(f'Q{i}') for i in range(1,13)),'Answer the original twelve research questions with measured evidence'
    q=pd.read_csv(ROOT/'results/qwen_metrics.csv');o=pd.read_csv(ROOT/'results/ouro_metrics.csv');m=pd.read_csv(ROOT/'results/math_metrics.csv');h=pd.read_csv(ROOT/'results/halting_selected.csv');a=pd.read_csv(ROOT/'results/actual_halting_latency.csv');p=pd.read_csv(ROOT/'results/trajectory_patterns.csv');s=pd.read_csv(ROOT/'results/absolute_stability.csv');geometry=pd.read_csv(ROOT/'results/projection_geometry.csv')
    selected=json.loads((ROOT/'configs/mc_selected.json').read_text());math_selected=json.loads((ROOT/'configs/math_selected.json').read_text());budget=json.loads((ROOT/'logs/gpu_budget.json').read_text());execution=json.loads((ROOT/'configs/math_execution.json').read_text())
    quality=[('method','Method','text'),('n','N','text'),('accuracy','Accuracy %','percent'),('delta','Δ pp','percent'),('ci_low','Δ CI low','percent'),('ci_high','Δ CI high','percent'),('wrong_to_correct','W→C','text'),('correct_to_wrong','C→W','text')]
    recovery_note=''
    recovery_path=ROOT/'logs/recovery_validation1p4_20261026/RECOVERY_SNAPSHOT.json'
    if recovery_path.exists():
        snapshot=json.loads(recovery_path.read_text());lineage=json.loads((ROOT/'results/recovery_lineage_check.json').read_text());assert lineage['passed']
        replay=json.loads((ROOT/'results/recovery_boundary_replay.json').read_text());assert replay['passed']
        recovery_note=f"- The first long Ouro1.4 validation process exited with native status0xC0000409 after {snapshot['valid_complete_records']} complete records. Windows WER identified ucrtbase.dll; its underlying cause remains unestablished. All saved records passed independent CPU consistency checks. Three boundary cases reproduced exact tokens/text; the baseline cases also matched their per-depth argmax; the next case passed decoding, numeric scoring and serialization, with a captured zero native exit. The resumed formal record matches that probe, and the original prefix is byte-identical. The entire failed supervisor interval ({snapshot['charged_failed_seconds']/3600:.4f} h) and both diagnostic runs were charged. Original logs/snapshot and the first probe's missing PowerShell child-exit metadata are retained. No inference kernel or completed record changed.\n"
    text=f'''# Looped / Training-Free Baseline Extension — seed 2021

This report adds a disjoint benchmark suite and generation/actual-exit checks to the sealed Stage1 audit. It does not replace the original report or reuse its test questions. All numbers below come from this extension's raw predictions.

## Research decision

**{decision['verdict']}**

{decision['reasoning']}

'''+ '\n'.join('- '+entry for entry in decision['evidence'])+'\n\n'+ '\n\n'.join(f'**Q{i}.** '+decision['question_answers'][f'Q{i}'] for i in range(1,13))+f'''

The decision concerns the proposed continuation-aware research direction, not permission to train or add a model. No training, fine tuning or learned continuation module was performed. These observations alone cannot establish that a hidden state is causally reusable: no state transplantation or counterfactual continuation intervention was run.

## Frozen scope and comparability

- Qwen3-4B-Base: 250 new validation questions; 600 new test questions, equally divided between ARC-Challenge, OpenBookQA and five MMLU subjects.
- Each Ouro model (1.4B and 2.6B): the same validation set and a nested 300-question test subset, 100 per benchmark family.
- Actual lm-eval 0.4.9 ConfigurableTask prompts: ARC 25-shot, OpenBookQA zero-shot, MMLU five fixed official-dev demonstrations. Primary metrics are acc_norm for ARC/OpenBookQA and acc for MMLU. Both summed and character-normalized scores remain in raw outputs. MMLU uses only the five listed subjects, not the full MMLU benchmark.
- GSM8K: 100 official-training validation questions, excluding the three fixed demonstrations; {execution['test_count']} fresh official-test questions, excluding all 100 old Stage1 test questions. The frozen maximum was 200. Actual count was fixed from measured pilot/validation runtimes before test generation; see configs/math_execution.json. Every model/method uses the same IDs, three demonstrations, greedy decoding and 512-token cap.
- Seeds and method grids were frozen before predictions. Hyperparameters are selected from validation only, separately per MC benchmark. Paper-fixed settings are reported independently. All grid results remain available.
- BF16 model computation, FP32 score math, no quantization. Same checkpoint within each paired comparison. Cross-model differences do not isolate architecture or scale causally.

This is a benchmark-subset reproduction with explicitly scoped adaptations; it does not reproduce the papers' full task suites or headline numbers.

## Source and implementation gates

1. [Training-Free Looped Transformers](https://arxiv.org/abs/2605.23872): the [official repository](https://github.com/L-z-Chen/Training-Free-Looped-Transformer) still returns 404 on webpage/API/raw/git checks. The author account API works but does not list this repository. Historical search-index README text is available; it is not a verifiable executable checkout. Qwen A1–A4 remain an independent recipe, and direct author-patch parity is unverified. A third-party independent reproduction is not treated as an official mirror.
2. [DoLa official code](https://github.com/voidism/DoLa/tree/{CFG['dola_commit']}): pinned source and AST scoring oracle were checked against real Qwen logits. The original implementation targets LLaMA; our Qwen run is an architecture adaptation. Intermediate states are raw, mature state is normalized, and the official KL direction, normalization-before-relative-top-mask order and mask −1000 are preserved. FP32 scoring is explicit. DoLa contrastive scores are not calibrated conditional NLL; their NLL table cells are blank.
3. [LoopCD](https://arxiv.org/abs/2610.02185): final-vs-first recurrent Logits contrast, fixed MC strength .5 and adaptive cap 1; generation fixed .3 and adaptive cap .5/1 for Ouro 1.4B/2.6B. Adaptive strength uses the final full-vocabulary top-two probability margin. Hidden contrast combines pre-output-norm states and reads the guided state through an output norm/head. The frozen generation collector also computes an unused mature output head before this guided readout; its runtime is not a zero-extra-head implementation benchmark. Ouro-Hidden is a formula-extension diagnostic; the paper's Ouro results used Logits. R2/R3 generation is an additional true-depth diagnostic rather than a claimed paper Ouro half-depth baseline.
4. [Adaptive Depth](https://arxiv.org/abs/2607.20519): 51 validation quantiles [.005,.995], native gate α=.01…1 and per-benchmark validation selection. Positive native thresholds only; no synthetic negative/infinite fallback. BF16 threshold rounding and native gate arithmetic are preserved.
5. Ouro public forwards, all four readouts, native exits, full/incremental caches, true 2/3/4-depth HF.generate and custom generation were parity checked. Independent HFLM likelihood checks cover ARC25/OpenBook0/MMLU5 for every checkpoint. Actual stopping additionally verifies decoder-layer counts, exit decisions and all candidate likelihoods on every latency question.

Sources, revisions and hashes are under sources/, configs/ and results/model_load_*.json. RELATED_WORK.md also compares the primary FlashLoop, LoopSpec and per-token convergence papers; those contextual methods are not claimed as reproduced here. The original 175 sealed Stage1 artifacts remain byte-identical.

## 1. Ordinary Transformer: training-free loops and DoLa

A0 is the original checkpoint. A1 repeats the four-layer window twice naively; A2 repeats it six times. A3 uses six damped Euler steps (step 1/6); A4 adds a .5 anchor to the natural single-pass block output. The window is layers [15,16,17,18]. There is no decoding phase in these likelihood requests. DoLa low/high pools are fixed adaptations to the 36-layer checkpoint; each benchmark's pool is selected using validation.

{table(q[q.task=='all'],quality)}

Δ compares each method with A0 on the same questions. NLL/entropy, independent complete-run walltimes and peak memory are in results/qwen_metrics.csv. Theoretical decoder-layer ratios are 1.111 for A1 and 1.556 for A2–A4; output-head/readout and system overhead are additional. DoLa performs no extra decoder passes but does add intermediate output heads. Entropy or NLL changes alone do not count as answer correction; W→C/C→W above measure changed correctness.

Per-benchmark DoLa selection: `{json.dumps(selected['qwen_dola'],sort_keys=True)}`.

![Qwen results](figures/figure6_qwen_training_free_baselines.png)

## 2. Trained looped Transformers: depth and decoding controls

The next table pools the same 300 questions per checkpoint. Guided and control scores are derived from the same four-loop per-candidate core computation. Those shared forward times cannot establish method-specific latency. Mean-centered parallel and orthogonal logit projections test whether the correction resembles a scalar reweighting of the final logits or a change in another direction; neither alone establishes a causal mechanism.

'''
    names=['r1','r2','r3','r4','logits_0.5','adaptive_paper','hidden_0.5','logits_validation_selected','hidden_validation_selected','temperature_validation_selected','parallel_05','orthogonal_05']
    for model in ['Ouro-1.4B','Ouro-2.6B']:
        text+=f'### {model}\n\n'+table(o[(o.model==model)&(o.task=='all')&o.method.isin(names)],quality)+'\n\n'
    text+='''Δ compares each row to that checkpoint's r4. W→C and C→W in quality tables always run from the reference method to the displayed row: A0 for Qwen, r4 for Ouro, and baseline for math. Thus an r1 row's W→C counts r4-wrong/r1-correct questions; chronological first-to-final transitions are reported separately below. Full strength/temperature grids, benchmark strata and MMLU subjects are in results/ouro_metrics.csv; all validation choices are in configs/mc_selected.json. Candidate score softmax weights describe relative candidate scores and are not calibrated answer probabilities.

Direct paired comparisons against validation-selected temperature, and Hidden-vs-orthogonal/Logits-vs-parallel controls, are in results/readout_pairwise_controls.csv. Overlap between two separate baseline-relative intervals is not used as a test of their difference. Positive scalar temperature leaves greedy next-token argmax unchanged, so it is a candidate-likelihood scoring control here rather than an additional distinct greedy-generation method.

![Decoding controls](figures/figure7_loopcd_paper_and_geometry.png)

### Correctness trajectories and reversals

'''+table(p[p.task=='all'],[('model','Model','text'),('pattern','Pattern','text'),('n','N','text'),('count','Count','text'),('rate','Rate %','percent'),('ci_low','Wilson low %','percent'),('ci_high','Wilson high %','percent')])+'''

Pairwise first/final classes partition the sample. Reversal classes anchor their final state at R4: C→W→C requires R4 correct and an earlier C→W pair among R1–R3; W→C→W requires R4 wrong and an earlier W→C pair. These reversal classes can overlap the first/final endpoint classes. The CSV preserves all benchmark strata and the raw records preserve the full four-step paths.

LoopCD changes are stratified by these endpoint/reversal classes in results/loopcd_trajectory_attribution.csv. Raw-vs-character-normalized scoring sensitivity is reported for every method in results/score_rule_sensitivity.csv, without changing the frozen primary scoring rule. Representative examples in results/representative_cases.json use the first frozen-manifest ID matching each benchmark/pattern, preserving question, choices, gold, all depths and paper-guided outputs. They illustrate observed trajectories rather than estimate prevalence.

results/depth_diagnostics.csv separates full-vocabulary and candidate-weight confidence/entropy/margins at each depth. results/trajectory_strata.csv reports patterns by question-character-length and initial full-vocabulary-confidence quartiles, using per-benchmark validation cuts. These are descriptive strata rather than new routing methods. Qwen window-state dynamics remain the separate sealed Stage1 audit; this extension does not claim new Qwen hidden-state evidence from its likelihood-only records.

## 3. Halting: validation policies, missed corrections and actual execution

Offline policy curves replay current/history-only decisions on the four-loop traces. They measure an accuracy–recurrent-depth tradeoff, not actual speed. The table below uses the predeclared validation average-depth target 3; policies that cannot meet a target are absent. The selected test depth can differ from the validation target.

'''+table(h[(h.task=='all')&(h.validation_budget==3)], [('model','Model','text'),('method','Policy','text'),('average_depth','Test mean R','float'),('accuracy','Accuracy %','percent'),('delta','Δ pp','percent'),('ci_low','Δ low','percent'),('ci_high','Δ high','percent'),('missed_final_corrections','Missed W→C','text'),('avoided_final_degradations','Avoided C→W','text')])+'''

All 13 metrics, every fixed validation grid threshold and budgets 1.5/2/2.5/3/3.5/4 remain in results/halting_*.csv. The matched-random control permutes depths within each benchmark while preserving its exact depth histogram (10,000 assignments). This distinguishes a useful allocation from merely running fewer loops, but is not a learned policy or a cross-domain generalization test.

The earliest-correct offline bound is in results/offline_oracle.csv. It reads gold/future correctness and is an analysis upper bound, never a deployable algorithm. Different policies' test mean depths need not match merely because their validation budgets match; use the full curves and each policy's exact-depth-histogram random control when comparing compute allocation.

![Frozen halting policies](figures/figure8_validation_frozen_halting.png)

### Absolute stability is descriptive

'''+table(s[(s.task=='all')&(s.metric=='full_vocab_kl')&(s.threshold==.05)], [('model','Model','text'),('depth','Observed R','text'),('n','KL≤.05 N','text'),('final_corrections','Later W→C','text'),('final_degradations','Later C→W','text'),('ci_low','Correction rate low %','percent'),('ci_high','Correction rate high %','percent')])+'''

KL=.05 is a descriptive fixed cut, not a universal convergence criterion. Hidden-relative-L2 cuts and other KL cuts are reported alongside it. Continuation AUC ranks currently wrong R2/R3 examples; the positive label is exactly a correct R4 answer. It was not used to tune policies and is not evidence of causal hidden-state preservation. AUC below .5 indicates an association in the opposite direction, not absence of information. Both raw/predeclared and reversed-sign AUC are preserved; neither orientation was chosen on test. This conditional-on-wrong diagnostic also uses a gold-defined subset that a deployable policy cannot identify directly.

Full-vocabulary KL is measured at the common prompt endpoint. It need not reflect changes at later positions in a multi-token candidate, whereas choice KL uses whole-candidate scores. A low endpoint KL followed by a changed MC answer therefore does not alone establish a failure to preserve useful continuation state.

### Actual early exit and measured latency

The norm/gate hooks interrupt the published forward after the selected recurrent step. No attention, mask or RoPE implementation is rewritten. First-candidate current-prefix signals choose the depth; all other candidates execute that true fixed depth. All 30 balanced test questions × five methods pass actual-layer-count, exit and likelihood parity checks before timing. Each timing uses five randomized repetitions; the independent bootstrap unit is the question, with its median repetition time.

'''+table(a[a.task=='all'],[('model','Model','text'),('method','Method','text'),('average_depth','Mean R','float'),('accuracy','Accuracy %','percent'),('mean_question_median_s','Seconds/question','float'),('speedup','Speedup','float'),('speedup_ci_low','Speedup low','float'),('speedup_ci_high','Speedup high','float')])+'''

These timings include policy readouts and complete pretokenized MC likelihood scoring with use_cache=False. They do not establish adaptive per-token generation, changing-depth KV reuse or serving speedup. Another user's GPU workload may share the device; randomized paired repetitions reduce temporal drift but do not provide an isolated-serving benchmark. Test quality at N=30 is a parity/latency subset result, not a replacement for N=300 quality.

![Actual stopping latency](figures/figure10_actual_halting_latency.png)

## 4. Autoregressive GSM8K: fixed, adaptive and selected decoding

'''+table(m,[('model','Model','text'),*quality,('numeric_accuracy','Numeric EM %','percent'),('truncated','Hit 512 cap','text'),('identity_reused','Identity reuse','text')])+f'''

Strict EM requires the extracted #### number to match the gold string after comma removal. Numeric EM separately uses Decimal equivalence. Missing/malformed extraction is incorrect. All raw generated text/token IDs, caps, actual cache levels and adaptive-strength ranges are preserved. A 512-token cap is a protocol limit, not proof that an incorrect answer could not be solved with a longer budget.

All nine validation conditions per checkpoint, including harmful or truncated high-strength guidance, are tabulated in results/math_validation_grid.csv. Validation outcomes choose a strength; they are not substituted for held-out test results.

Validation selection: `{json.dumps(math_selected,sort_keys=True)}`. Zero-strength selected methods reuse the identical baseline output explicitly; identical selected/fixed Hidden .25 outputs are also reused. Reuse is recorded, rather than pretending those are independent generations.

Baseline-only extra heads record all-depth per-token predictions, KL and hidden changes along the same generated prefix. Agreement with the final-loop next token is **not gold-token correctness**. These token traces differ from both the MC same-prefix depth audit and separately generated fixed-depth paths. Baseline tracing adds output-head/synchronization overhead; audited generation runtimes must not be presented as clean method speedups. Raw costs are retained for budget accounting.

![Generated-answer results](figures/figure9_actual_autoregressive_generation.png)

## Statistics, resources and reproducibility

- Paired accuracy intervals: 10,000 bootstrap question resamples, seed 2021. Correctness-transition and trajectory rates use Wilson intervals; transition-rate denominators are all aligned questions, not only initially wrong/correct questions. Counts and all four rates/intervals are retained in the CSVs. Latency uses a paired question bootstrap of median repeat costs. Intervals are pointwise and unadjusted for the many reported comparisons; crossing zero is not proof of no effect or noninferiority.
- Cumulative charged GPU walltime: {budget['used_seconds']/3600:.4f} h, including sealed Stage1 {budget['stage1_used_seconds']/3600:.4f} h and all extension failures/pilots/loading. The ledger's hard cap is 50 h. This is elapsed exclusive experiment-owner walltime, not hardware profiler active-kernel time.
- Memory values use torch.cuda.max_memory_allocated. Qwen peaks cover complete method runs; shared Ouro MC peaks cover the common readout graph, not independent methods. Generation records are process high-water upper bounds; actual-halting measurements reset the allocator peak per timed condition.
- Validation and all control methods are retained, including harmful high-strength guidance, source failures and the recovered Windows long-path environment install failure. Original error logs remain. Short D-drive environment: `{CFG['environment_python']}`.
- At the user's request, the extension-added Ouro-2.6B snapshot resides on E through an NTFS junction at its original D model path. All 12 copy hashes and the pinned weight LFS hash were verified. Scripts, results and the environment remain on D. See STORAGE_LAYOUT.md and logs/storage_migration_20261026.json; this relocation changed no inference code or model bytes.
- The first CPU MC analysis exited with native status 0xC0000409 after all GPU scoring had passed. The same calculations completed successfully with added progress/faulthandler diagnostics. Its root cause is unestablished. An independent 18-file raw-output audit gated recovery; original failure/exit logs remain, and no GPU prediction was rerun for that recovery.
{recovery_note}
- Completion audit independently rederives selections from validation, checks raw prediction consistency, source/protocol/implementation hashes, sample alignment, actual stopping, generation parity and the old Stage1 seal. See results/completion_audit.json.
- Reports/tables/figures can be rebuilt on CPU from raw JSONL. GPU reruns require the original Stage1 package and pinned local model/dataset snapshots. See REPRODUCE.md; weights/cache/environments are excluded from delivery.

The original Stage1 zero-shot/100-question generation results are separate historical evidence. Do not pool their rates with this new few-shot/disjoint suite or interpret the combined size as one identically specified experiment.
'''
    (ROOT/'FINDINGS.md').write_text(text,encoding='utf-8',newline='\n')
    save('results/report_provenance.json',dict(completion_audit_passed=True,decision=decision,sources='local audited CSV and raw JSONL',no_paper_reported_numbers_substituted=True))
    print(json.dumps(dict(report='FINDINGS.md',bytes=len(text.encode('utf-8')))),flush=True)

if __name__=='__main__':main()
