"""Pure CPU descriptive eight-corner metrics; no model or data access."""
import math,statistics
if not __debug__:raise RuntimeError('optimized Python forbidden')
NAMES=('111','112','121','122','211','212','221','222')
def evaluate(v):
 if set(v)!=set(NAMES) or any(not isinstance(v[k],(int,float)) or not math.isfinite(v[k]) for k in NAMES):raise ValueError('corners')
 r={}
 for k in ('1','2'):
  m11=v['11'+k];m12=v['12'+k];m21=v['21'+k];m22=v['22'+k]
  r['P_Q2_R'+k]=m22-m12;r['P_Q1_R'+k]=m21-m11;r['Q_P2_R'+k]=m22-m21;r['Q_P1_R'+k]=m12-m11;r['interaction_R'+k]=m22-m12-m21+m11;r['total_C_R'+k]=m22-m11
  r['path_P_then_Q_roundoff_R'+k]=(m22-m11)-((m22-m12)+(m12-m11));r['path_Q_then_P_roundoff_R'+k]=(m22-m11)-((m22-m21)+(m21-m11))
 r['interaction_R2_minus_R1']=r['interaction_R2']-r['interaction_R1'];return r

def describe(xs):
 if not xs:return {'n':0,'mean':None,'median':None,'positive':0,'negative':0,'zero':0}
 return {'n':len(xs),'mean':statistics.mean(xs),'median':statistics.median(xs),'positive':sum(x>0 for x in xs),'negative':sum(x<0 for x in xs),'zero':sum(x==0 for x in xs)}
