from supervisor_draft import check,GIB
s={'gpu':{'free_mib':21*1024},'disk_free_bytes':21*GIB,'memory':{'MemAvailable':9*GIB}}
check(s,0,True);check(s,1)
for k in ('gpu','disk','memory','time'):
 x={'gpu':dict(s['gpu']),'disk_free_bytes':s['disk_free_bytes'],'memory':dict(s['memory'])};elapsed=1
 if k=='gpu':x['gpu']['free_mib']=1
 elif k=='disk':x['disk_free_bytes']=1
 elif k=='memory':x['memory']['MemAvailable']=1
 else:elapsed=7201
 try:check(x,elapsed)
 except RuntimeError:pass
 else:raise AssertionError(k)
print('SUPERVISOR_RESOURCE_TOY_PASS; NO GPU')
