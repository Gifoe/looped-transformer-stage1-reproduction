import sys,pathlib
P=pathlib.Path('/home/luansiyu/loop-latent');sys.path.insert(0,str(P/'repo/experiments/ouro_latent_state_mechanism_v1'))
from pqr_supervision import stage_admissible
G=1024**3
s={'gpu':{'free_mib':21*1024},'disk_free_bytes':40*G,'memory':{'MemAvailable':20*G}}
assert stage_admissible(s,20*G,G)
assert not stage_admissible(s,max(20*G,18*G+5*G),G)
s['gpu']['free_mib']=24*1024;assert stage_admissible(s,23*G,G)
print('CURRENT_FREE_VS_MEASURED_PEAK_CPU_TEST_PASS')
