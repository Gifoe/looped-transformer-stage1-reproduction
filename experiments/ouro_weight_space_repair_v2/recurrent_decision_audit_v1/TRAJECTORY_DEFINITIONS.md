# Pre-execution trajectory definitions

For real native K1,K2,K3,K4 readouts, independently for original joint and cyclic mean:
- first correct: minimum correct K (retrospective; missing if never correct).
- first answer change: minimum K>=2 differing from previous K.
- first local stable confirmation: minimum K>=2 equal to previous K; stable start is K-1.
- persistent stable suffix start: first K in 1..3 whose suffix through K4 is constant (requires at least two depths); missing otherwise.
- later reversals: adjacent answer changes strictly after first local stable confirmation; this is not necessarily return to the same earlier answer.
- also report all adjacent changes and returns to any previously abandoned answer separately.

For historical test records only K1,K4 exist. Equal available endpoints are confirmed at K4 with gap3; K2/K3 reversals and exact first correction are unknown. No interpolation. Permutation stability means agreement of content predictions after displayed-order argmax, with first displayed label winning exact ties. Cyclic combined scores use historical arithmetic mean of already mapped complete letter-target LP sums and original-content argmax. No answer-body character division is applied to joint letter sums.
