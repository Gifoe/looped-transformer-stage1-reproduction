# Native block propagation, generation and competing effects

Fix the entire native block map F_l at the four typed prefixes. All calls share actual no-cache masks and rotary parameters. Let s=h10+h01-h00 be the additive reference residual. The two terms at the final valid position are

```text
Incoming interaction after the block: F(h11)-F(s)
New interaction in the block:          F(s)-F(h10)-F(h01)+F(h00)
Total output interaction:             F(h11)-F(h10)-F(h01)+F(h00)
```

Incoming interaction can be transformed nonlinearly. The division depends on the neural coordinates and chosen additive reference. In Ouro each native block is h+A(h)+M(h+A(h)), where A and M are the normalized attention and MLP residual writes. Because the additive inputs cancel, the generated piece is exactly the sum of the two writes' mixed differences. The verifier requires replay of native module outputs and the two reconstructions. The applied single-site intervention leaves the original block and all earlier states intact and changes only its outgoing readout to remove or amplify the chosen component. Later blocks and subsequent tokens run natively. Changing the actual attention write first is a distinct operator-order test.

For the norm sensitivity, radially clip s to the range of the four native norms at every token. Put the clip's additive input contribution into the propagation term. The new-generated piece then continues to consist of attention and MLP mixed writes. This verifies how robust the decomposition is to a norm restriction. It does not certify the semantic validity of the interpolated full state.

The geometric control preserves the intervention norm and its projection onto both original K1 readout and the K2-minus-K1 readout at that site. Those controls keep the resulting readout norms fixed and perturb the remaining direction. The amplitude-matched propagation intervention tests a competing direction. The same-answer safety comparison includes native confidence matching and an exact same-vector, same-dose transfer. Matching uses absolute K1 margin because sign equality would exclude the native K1 error-versus-correct comparison. Signed K2 margins can be matched. For a deliberately fixed selection rule, the discovery direction is K2's diagnostic score difference versus matched geometry. Actual K1 and K2 outcomes remain held out until the selected layer is frozen.

Norms and score changes are descriptive. The decision endpoint is the freely completed number, compared to the original exact y or y+100 in native repairs. Tested operations apply at one pivot and do not identify a unique symbolic implementation. [Path patching](https://arxiv.org/abs/2304.05969) · [Arithmetic mechanisms](https://arxiv.org/abs/2402.02619)

