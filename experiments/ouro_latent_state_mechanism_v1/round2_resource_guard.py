"""Pure pre/post resource assertions for a completed supervised probe."""
GIB=1024**3
def check_probe_resources(pre,post,peak_reserved):
 if post['gpu']['compute_pids'] or post['disk_free_bytes']<20*GIB or post['memory']['MemAvailable']<4*GIB:raise RuntimeError('post-probe resource conflict')
 if peak_reserved+4*GIB>pre['gpu']['free_mib']*1024**2:raise RuntimeError('insufficient GPU headroom')
