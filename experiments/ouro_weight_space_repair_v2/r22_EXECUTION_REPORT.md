# R22 complete576: trailing relative-step value signal failed

All576 questions and every cached candidate were processed by the preregistered gold-free CPU worker before label analysis. The supervisor completed cleanly in18.70353743294254seconds; feature processing13.362517293076962seconds, including2,291,846,842bytes of cache reads and SHA verification. The one subsequent frozen analysis completed in1.219seconds. No model/weights/new forward or CUDA operation was used. Tokenizer loading was pinned/local-only for input indexing audits. New GPU-worker seconds0; original ledger total18472.29435229581 and balance3127.7056477041915 unchanged.

All state SHA/ID/candidate/target/position/shape checks passed. Every question had identical independently encoded prefix IDs across candidates and exact first-position states at all six native depths. First real NumPy/Torch FP64 discrepancy was1.1102230246251565e-16, within the frozen1e-12 threshold. Degenerate denominator count0/13417positions. Every both-single-token challenge had deltaV exactly0, as predicted by the shared first state.

| Scope | Rescue questions/pairs | Damage questions/pairs | Main V AUC |
|---|---:|---:|---:|
| Discovery |38/38|129/144|0.542738|
| Selection |16/16|65/72|0.476442|
| Combined |54/54|194/216|0.520809|

The54 rescue IDs exactly matched R21's discrete-reachable R6 errors. Each question had total weight1 within its primary class; background both-wrong pairs were excluded from primary AUC. The main combined95% percentile CI was[0.434659,0.607342], using10000 seed2021 rescue/damage-stratified question bootstrap draws with the same indices for all five signals.

Combined controls: negative absolute energy AUC0.615263, native6 score0.705183, log-token length0.560877, negative log-token length0.439123. Main minus native6 AUC difference=-0.184374 with paired95%CI[-0.301699,-0.063902]. Main minus energy=-0.094454, CI[-0.191724,0.003007]; main minus positive length=-0.040068, CI[-0.158983,0.079976]; main minus negative length=0.081687, CI[-0.037042,0.199362]. Controls are ineligible for promotion.

The power requirement passed, but both split AUC>=0.65, combined CI lower>0.5 and all four paired control-advantage conditions failed. The specific hypothesis that larger trailing relative step-decay is a useful correctness value signal fails its frozen gate. No sign change, window/norm search, controller, feature combination, threshold or subgroup selection is authorized. This does not establish that every possible stability feature is useless.

Descriptive coverage (question/pair counts per rescue,damage,background): all54/54,194/216,89/94; equal-token20/20,93/99,39/39; both-single6/6,22/24,9/9. Background and rescue categories can share a question; these are not disjoint question totals. No subgroup AUC or additional significance test was run, and no subgroup was filtered into the primary analysis.

This reused development association study is neither independent validation nor a causal/deployable decoder. The original54 oracle opportunities remain a capacity fact, not achieved rescues. No confirmation517/validation299/test question records, other datasets or external benchmarks were accessed. The research goal remains unachieved.

Definition89a8a34cc4916f9f833e3b35585867020d5b5f07; protocol2dad20de948fbe73411de89fc5544168e279236c24e1bd30862448cd8742b09d; complete featuresSHA0749ea1c489c7f9a32be51d15b6e297b937b025538314f5f8522488aa8aeafdf.
