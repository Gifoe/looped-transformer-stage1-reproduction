# Conditional E7 actual runtime profile — prepared, not run

Source: `/root/autodl-tmp/ouro-e7/reencoding/E7_ACTUAL_PROFILE.py`, SHA256 `360acbc3b1294afc31f6279b7623b66fd34608d6431beadca353dcb64a66c272`.

Prepared sample/permission freeze: `actual_profile/PROFILE_SAMPLE_AND_SOURCE_FREEZE.json`, SHA256 `9eef4cf9c2aadfbef06d4bb076bc9ed441b4dbc02bad19c5f87785778ce1b3ad`.

Runtime input: `actual_profile/PROFILE_GOLDFREE_TRAIN128.jsonl`, SHA256 `4ccda361c34cf74d3fceed9124f74b745cc6f2be20dba119aedf51846774642e`.

All128 are original four-choice Train questions, one lexical representative per original input group, selected by SHA256(`E7-runtime128-20261113:` + group). Selection reads only original id/group/input-hash metadata and gold-free variant-input fields, never new outcomes. First two fixed technical IDs: `Mercury_7168683`, `TIMSS_1995_8_O11`. Query gold is absent from the runtime fixture and worker; the original fixed25 demonstration answers remain in prompt text as required.

Only `--phase prepare` was executed. No GPU action occurred. Current collector/analysis/queue sources were not modified.

Root alone may invoke, after scientific review:

```sh
/root/autodl-tmp/ouro-e6a/venv/bin/python -u /root/autodl-tmp/ouro-e7/reencoding/E7_ACTUAL_PROFILE.py --phase technical
/root/autodl-tmp/ouro-e6a/venv/bin/python -u /root/autodl-tmp/ouro-e7/reencoding/E7_ACTUAL_PROFILE.py --phase profile
```

The worker first checks the frozen Train decision. If neither pair nor selected-gate criterion passed, it writes `PROFILE_NOT_RUN.json` before importing torch or touching GPU. Any eligible GPU phase additionally requires full22280 completion and matching frozen Validation review. Full128 requires the same-source/sample/Train/Val first-two technical `COMPLETE.json` PASS. It acquires the existing nonblocking GPU lock and never steals/restarts another task.

Policies are nativeK4, original strong output calearly .6/entropy.5, and only the eligible fixed pair and/or selected gate. Every policy retokenizes its actual prompt(s), checks IDs, constructs CUDA tensors, runs real control flow and all required readout/CPU/aggregation work, then synchronizes. Two K2 forwards are executed for the pair. Gate uses the actual gate output once per loop and padded4-column prefix PDF/CDF without future gates. Strong stops from the norm callback and does not compute the current loop's otherwise unused gate after throwing.

Counter expectations: native4 =1body/4first-layer/1head/4gate; pair =2body/4first-layer/2head/4gate; strong stopping atd =1body/d first-layer/d heads/(d−1) gate; gate stopping atd =1body/d first-layer/1head/d gate. Endpoint full-vocabulary SHA, frozen replay decisions, pair FP64 scores, and prefix BF16 CDF equality are checked outside both timers. Small R-only QA clones/scalar observer hooks remain inside measured windows and are disclosed. Cold first unit/all first-question policy units are retained separately; policy order cycles by question.

Outputs keep raw timings, end-to-end and model/readout/decision windows, model/tokenizer/import setup times, policy-order/actual counts, GPU peaks, endpoint checks and parameter versions. No accuracy, fitted head, new threshold, Test data, or declaration of Goal success is produced by this measurement worker.
