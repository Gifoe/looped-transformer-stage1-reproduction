# Causal specification and scope

Let F be one native shared pass including the output RMS normalization and let H_1 = F(E(tokens)). Let P select the graft's contiguous input positions. Given the recipient A+B and donor A*+B*, set

```text
H_1^hybrid = (I - P) H_1(A+B) + P H_1(A*+B*)
H_2^hybrid = F(H_1^hybrid)
```

For each greedy generation step append the recipient's generated prefix to its token input. The donor cache is valid at the graft because its positions precede both B* and every generated token. Consequently the graft's source hidden states are functions only of the common demonstrations, A*, and the intervening syntax. The final position used by direct H_1 unembedding is outside P. The assay verifies bitwise equality of that state after the graft, and equality of full and resumed native K2 under the same tensor shape. The three mutually exclusive exact outputs are A+B, A*+B, and A*+B*. Selecting A*+B operationalizes counterfactual dependence on both the donor's first operand and the recipient's remaining context. It is compatible with recomputation or with modifying candidate information already present on the recipient's trajectory.

For the geometry control, let r be the native recipient vector, d the donor, and u the unit vector in direction r. Preserve d·u. Replace the perpendicular part of d by an independent random perpendicular direction with the same norm. The control preserves both ||d|| and ||d-r|| at the exact hooked position. Since this is a geometric control, conclusions beyond it still depend on the counterfactual, two directions, and disjoint value combinations.

For a location s in the graft, define the attention write at a query j via softmax over Q_j K_i^T for its causally permitted source positions i. The value term is the weighted sum of V_i. To block graft information at all suffix positions, restore recipient K_s and V_s for the graft positions in every physical K2 layer. Inducting through the layers, query states outside the graft start native. Inputs available outside the graft are then native, the only changed K and V were restored, and the MLP and normalization are tokenwise. Thus each subsequent outside-graft state remains native, up to numerical precision. Individual K and V restorations test different projections while allowing all other affected paths to interact. Their effects can be nonadditive and apply to the intervened paths as a whole. Interchange intervention is prior methodology. [Geiger et al., Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997)

The behavioral split, the operand-state assay, and the attention restoration have distinct scopes. The first tests spontaneous exact same-output collisions, the second tests whether prior-state information is used in a second loop, and the third localizes routes within an experimentally induced causal change. The third's confirmation population is conditioned on the second having a two-way prefix effect. The assay tests only the fixed checkpoint, demonstrations, digit length and hundreds-place changes. Findings about carry algorithms, other task families, and globally sufficient explanations require their own causal evidence. [Ouro architecture and training](https://arxiv.org/abs/2510.25741) · [Prior circuit analysis of addition and subtraction](https://arxiv.org/abs/2402.02619)

