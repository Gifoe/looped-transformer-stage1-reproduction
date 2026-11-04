# Frozen R6 ARC changed cases

Posthoc decomposition of the selected frozen rule. These cases were not used to change coefficients or choose another method.

All changes versus A1: {'unchanged': 1124, 'rescue': 23, 'damage': 13, 'wrong_to_wrong': 12}. Complete vectors: [candidate CSV](slcd_r6_arc_candidate_scores.csv); all changed cases: [JSONL](slcd_r6_arc_changed_cases.jsonl).

Of 35 rescues versus Fixed, 10 recover damage to an originally correct Native K4 answer and 25 correct questions both Native K4 and Fixed missed. Of 16 damages, 6 undo a useful Fixed rescue and 10 damage an answer both Native K4 and Fixed had correct. This is posthoc correctness accounting, not a causal mechanism.

The score intervention establishes which ranking component flipped a decision. It does not identify a hidden-state causal mechanism, factual knowledge source, or calibrated probability.

## First rescue in source order: arc_challenge/test/Mercury_7175875

An astronomer observes that a planet rotates faster after a meteorite impact. Which is the most likely effect of this increase in rotation?

Gold 2; A1 3; R6 2; alpha=1. Indices are zero-based.

| Choice | Text | A1 | Prior guidance e | Correction -alpha*e | R6 |
|---|---|---:|---:|---:|---:|
| 0 | Planetary density will decrease. | -0.505901 | -0.127572 | 0.127572 | -0.378329 |
| 1 | Planetary years will become longer. | -0.755555 | -0.182742 | 0.182742 | -0.572814 |
| 2 | Planetary days will become shorter. | -0.447707 | -0.177051 | 0.177051 | -0.270655 |
| 3 | Planetary gravity will become stronger. | -0.394045 | -0.118958 | 0.118958 | -0.275087 |

Gold margin -0.053662 -> 0.004431.

## First damage in source order: arc_challenge/test/Mercury_407327

A physicist wants to determine the speed a car must reach to jump over a ramp. The physicist conducts three trials. In trials two and three, the speed of the car is increased by 20 miles per hour. What is the physicist investigating when he changes the speed?

Gold 3; A1 3; R6 2; alpha=0.993172. Indices are zero-based.

| Choice | Text | A1 | Prior guidance e | Correction -alpha*e | R6 |
|---|---|---:|---:|---:|---:|
| 0 | the control | -1.255224 | -0.285178 | 0.283231 | -0.971993 |
| 1 | the hypothesis statement | -0.939934 | -0.175023 | 0.173828 | -0.766106 |
| 2 | the dependent (responding) variable | -0.547476 | -0.227690 | 0.226135 | -0.321341 |
| 3 | the independent (manipulated) variable | -0.532406 | -0.202049 | 0.200670 | -0.331736 |

Gold margin 0.015070 -> -0.010395.

