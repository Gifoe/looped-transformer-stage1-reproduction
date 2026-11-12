UTC: 2026-11-11T10:41:53Z

Astra independently reviewed and approved the three-case L18 probe for continuation to full cohort. A/B/C source, R and input identities, all eight BF16 full-vocabulary logits, natural B1/B2 and identity bitwise anchors, actual BF16 norm controls and independent CUDA score replay were verified. Scientific core and conditions remain fixed. The 40 GiB probe peak resulted from explicit preallocation and is not the actual requirement; the subsequent full run uses GPU memory only as needed under the user's updated resource instruction.

Probe receipt SHA256: 10be811e8fa43274f199dd52c870b53cdea8dfee6aadb91345812bacfe2d1d3a
Pending SHA256: fddf4e0bb993e5718a408bedac7ee07eba851e555fe4c53fb0722f566a95a2ef
CUDA replay SHA256: 317e6a4be520bf30c0397a0047444b5221e2175d460ccf8418ee6e772acc6cb1
