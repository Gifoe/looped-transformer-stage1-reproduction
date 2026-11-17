import pathlib,json
R=pathlib.Path(__file__).resolve().parent.parent;s=json.loads((R/'REGION_SELECTION.json').read_text());target=json.loads((R/'DEVELOPMENT_NONADDITIVITY.json').read_text())['graph_strata']
for k in target:
 e={int(m):p['four_strata'][k] for m,p in s['profiles'].items()};q={'E_Q':e[1],'E_O':e[2],'E_R':e[4],'E_QO':e[3],'E_all':e[7],'QO_nonadditivity':e[3]-e[1]-e[2],'QO_R_nonadditivity':e[7]-e[3]-e[4],'three_way_nonadditivity':e[7]-e[3]-e[5]-e[6]+e[1]+e[2]+e[4]};assert q==target[k]
print('POSTHOC_DEV_NONADDITIVITY_REPLAY_PASS (descriptive only)')
