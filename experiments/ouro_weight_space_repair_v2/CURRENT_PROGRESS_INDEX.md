# Current progress checkpoint

This snapshot uploads existing attempts, including failures and incomplete runs. The original research goal remains unachieved. No experiment was run for this upload.

## Navigation

Use [the machine manifest](artifacts/CURRENT_PROGRESS_MANIFEST.json) for each local file SHA, public direct/recovery coverage, exclusions and release indexes. Existing archives remain unchanged; public INDEX/CONTENTS files identify lossless recovery members. The new prepared source history is not an additional executed variant.

| Stage | Actual state | Code / results |
|---|---|---|
| R7 | Completed original stage; code, results and partial/cancelled branches retained. | `scripts/r7*`, `results/r7*`; matching artifacts indexes in manifest |
| R8 | Closed diagnostic. | `scripts/r8*`, `results/r8*`; matching artifacts indexes in manifest |
| R9 | Closed pilot, including initial failed attempt. | `scripts/r9*`, `results/r9*`; matching artifacts indexes in manifest |
| R10 | Closed development attempt. | `scripts/r10*`, `results/r10*`; matching artifacts indexes in manifest |
| R11 | Completed audited stage; no new claim in this checkpoint. | `scripts/r11*`, `results/r11*`; matching artifacts indexes in manifest |
| R12 | CPU signed diagnostics completed. | `scripts/r12*`, `results/r12*`; matching artifacts indexes in manifest |
| R13 | Prefix diagnostics completed. | `scripts/r13*`, `results/r13*`; matching artifacts indexes in manifest |
| R14 | Development scores completed. | `scripts/r14*`, `results/r14*`; matching artifacts indexes in manifest |
| R15 | Observational diagnostics completed; v0 history retained. | `scripts/r15*`, `results/r15*`; matching artifacts indexes in manifest |
| R16 | Native KL pilot failed gate. | `scripts/r16*`, `results/r16*`; matching artifacts indexes in manifest |
| R17 | Paired-reference pilot failed gate. | `scripts/r17*`, `results/r17*`; matching artifacts indexes in manifest |
| R18 | Uniform metric pilot failed gate. | `scripts/r18*`, `results/r18*`; matching artifacts indexes in manifest |
| R19 | Last-loop protection pilot failed gate. | `scripts/r19*`, `results/r19*`; matching artifacts indexes in manifest |
| R20 | V0 19/24 incomplete; E1 23/24 incomplete; E2 full24 failed gate, all costs retained. | `scripts/r20*`, `results/r20*`; matching artifacts indexes in manifest |
| R21 | CPU convex-hull capacity oracle completed; diagnostic only. | `scripts/r21*`, `results/r21*`; matching artifacts indexes in manifest |
| R22 | CPU energy diagnostic completed, gate failed. | `scripts/r22*`, `results/r22*`; matching artifacts indexes in manifest |
| R23 | SLED-inspired pilot completed, 0 rescue / 1 damage vs R6. | `scripts/r23*`, `results/r23*`; matching artifacts indexes in manifest |
| R24 | CPU demo-support admission completed; opportunity only. | `scripts/r24*`, `results/r24*`; matching artifacts indexes in manifest |
| R25 | Memory pilot completed, strong and cost flags false. | `scripts/r25*`, `results/r25*`; matching artifacts indexes in manifest |
| R26 | Finite prefix causality acceptance completed; 6/42 length pairs nonexact. | `scripts/r26*`, `results/r26*`; matching artifacts indexes in manifest |
| R27 | Artificial lookup intervention completed; V functional diagnostic passed, no ARC gain claim. | `scripts/r27*`, `results/r27*`; matching artifacts indexes in manifest |
| R28 | Spatial source intervention completed; strong and cost flags false. | `scripts/r28*`, `results/r28*`; matching artifacts indexes in manifest |
| R29 | Canonical proposal only, unexecuted. | `scripts/r29*`, `results/r29*`; matching artifacts indexes in manifest |
| R30 | Temporal source-clock pilot completed; strong and cost flags false. | `scripts/r30*`, `results/r30*`; matching artifacts indexes in manifest |
| R31 | CPU shared-prefix binary capacity oracle: 13 reachable, 11 unreachable; no method gain. | `scripts/r31*`, `results/r31*`; matching artifacts indexes in manifest |
| R32 | Natural-field interaction proposal only, unexecuted. | `scripts/r32*`, `results/r32*`; matching artifacts indexes in manifest |
| R33 | Common-gain pilot completed; 0 rescue / 2 damage vs R6, strong and cost false. | `scripts/r33*`, `results/r33*`; matching artifacts indexes in manifest |
| R34 | Design only; implementation and CPU checks not started; no GPU. | `scripts/r34*`, `results/r34*`; matching artifacts indexes in manifest |

## Resource and privacy scope

19 GPU-worker ledgers total **20,834.389865819365 / 21,600 seconds**, remaining **765.6101341806352 seconds**. CPU-only diagnostics do not consume a new GPU ledger. Components and total fields are not double counted.

Credentials, transport coordination, private host/resource identifiers, model weights, private tokenizer/upstream source copies and regenerable caches are excluded. The generic SSH template uses environment-supplied host/user and stdin password; the actual connection profile remains private. R34 is design-only. Unexecuted R29/R32/R34 proposals are not results.

The R23 postrun unfrozen review is a separately named postrun artifact and does not replace its frozen preregistered review. Historical v0 metadata is retained with its original bytes and does not replace final release definitions.
