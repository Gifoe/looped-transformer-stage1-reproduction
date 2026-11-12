# Q-key producer mediator scope clarification, before new GPU effects

This adds a precise mediator definition to ORDER_E3_QKEY_PRODUCER_v1.md (SHA256 0b880036365964e3f7527c0fee7081b543ce2a5452d6510b3f6aa7f602fedcdb). It does not add conditions or change cuts, IDs, source states, scoring or selection rules.

The mediator is the contribution of Q-region keys to the L18 **R query row**. Compute the original full-shape SDPA twice when necessary, once for the receiver background and once with K_Q exchanged. Write only the latter's R output row into the original SDPA output. Every non-R output query row stays bitwise equal to that receiver background. Subsequent o_proj/local norm, residual, MLP and later layers run normally.

Consequently, G01 transports the producer's K_Q through the L18 R-attention edge into clean natural K2. G10 restores clean K_Q only for the R-attention edge in the perturbed producer background. It does not restore the complete K_Q node's influence on other query positions or on their later-layer effects. Those consequences remain part of the background. Report this distinction in code receipts, analysis and causal maps.

Capture the actual cut input at ut1 before its Q splice, including complete tensor byte hash, shape/dtype, Q-region and R-region hashes and the actual injected-Q hash. Verify it against the same-question unperturbed capture at that cut. A textual source declaration is not this check. Complete hidden inputs may remain transient; do not claim offline reconstruction of arrays not archived.

The endpoint/source/identity gates and four-corner contrasts remain as specified in the parent order. An effect or restoration here is evidence about a restricted edge under the selected receiver background, not the entire key node or a complete natural mechanism.
