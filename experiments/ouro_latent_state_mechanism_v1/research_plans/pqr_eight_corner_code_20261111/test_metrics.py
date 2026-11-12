from pqr_metrics import evaluate,describe,NAMES
v={'111':0.,'112':1.,'121':2.,'122':4.,'211':3.,'212':6.,'221':8.,'222':12.};r=evaluate(v)
assert r['P_Q2_R2']==8 and r['P_Q1_R2']==5 and r['Q_P2_R2']==6 and r['Q_P1_R2']==3 and r['interaction_R2']==3
assert r['total_C_R2']==11 and r['path_P_then_Q_roundoff_R2']==0 and r['path_Q_then_P_roundoff_R2']==0
assert describe([-1,0,1])=={'n':3,'mean':0,'median':0,'positive':1,'negative':1,'zero':1}
for bad in ({**v,'111':float('nan')},{k:x for k,x in v.items() if k!='211'}):
 try:evaluate(bad)
 except ValueError:pass
 else:raise AssertionError('bad corner set accepted')
print('PQR_METRICS_SYNTHETIC_CPU_PASS')
