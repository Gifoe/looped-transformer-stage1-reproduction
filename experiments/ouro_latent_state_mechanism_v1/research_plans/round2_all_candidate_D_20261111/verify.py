import json,gzip,hashlib,pathlib,statistics,os
T=pathlib.Path(os.environ['TEMP']);o=T/'ouro-all-candidate-D-20261111';entries=[json.loads(x) for x in (o/'question_metrics.jsonl').read_text(encoding='utf8').splitlines()];by={(r['split'],r['id']):r for r in entries};assert len(by)==len(entries)==1418
for split,d,name in [('train','ouro-round2-train-disclosure-20261111','train_scores_compact.jsonl.gz'),('validation','ouro-round2-validation-prospective-20261111','validation_scores_compact.jsonl.gz')]:
 for line in gzip.open(T/d/name,'rt',encoding='utf8'):
  j=json.loads(line);r=by.pop((split,j['id']));g=j['gold'];s21=j['conditions']['K2C_K1R']['scores'];s12=j['conditions']['K1C_K2R']['scores'];v={str(k):(s21[g]-s21[k])-(s12[g]-s12[k]) for k in range(len(s21)) if k!=g};assert r['candidate_D']==v and r['all_candidate_mean_D']==statistics.mean(v.values()) and r['fixed_wrong_D']==v[str(j['fixed_wrong'])] and r['metadata_sha256']==j['metadata_sha256']
assert not by
print('ALL_1418_PER_CANDIDATE_VALUES_INDEPENDENTLY_RECOMPUTED')
