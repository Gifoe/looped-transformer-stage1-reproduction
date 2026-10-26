"""Read-only Windows CPU-set topology; does not change any process affinity."""
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
from pathlib import Path
import struct

root = Path(__file__).resolve().parents[1]
kernel = ctypes.WinDLL('kernel32', use_last_error=True)
query = kernel.GetSystemCpuSetInformation
query.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                  wintypes.HANDLE, wintypes.DWORD]
query.restype = wintypes.BOOL
length = wintypes.DWORD()
query(None, 0, ctypes.byref(length), None, 0)
assert length.value > 0
buffer = ctypes.create_string_buffer(length.value)
assert query(buffer, len(buffer), ctypes.byref(length), None, 0), ctypes.get_last_error()
data = buffer.raw[:length.value]
rows, offset = [], 0
while offset < len(data):
    size, kind = struct.unpack_from('<II', data, offset)
    assert size >= 8 and offset+size <= len(data)
    if kind == 0:
        assert size >= 32
        uid, group, logical, core, cache, node, efficiency, flags = struct.unpack_from('<IH6B', data, offset+8)
        rows.append(dict(id=uid, group=group, logical=logical, core=core,
                         cache=cache, numa=node, efficiency_class=efficiency, flags=flags))
    offset += size
assert len({(r['group'], r['logical']) for r in rows}) == len(rows)
classes = {}
cores = {}
for row in rows:
    classes.setdefault(str(row['efficiency_class']), []).append(row['logical'])
    cores.setdefault(f"{row['group']}:{row['core']}", []).append(row['logical'])
result = dict(checked_utc=datetime.now(timezone.utc).isoformat(), cpu_sets=rows,
              efficiency_classes=classes, logical_processors_by_core=cores,
              single_thread_core_logical_ids=[ids[0] for ids in cores.values() if len(ids) == 1],
              scope='Read-only topology. No process affinity, power, clock, firmware or other workload was changed.',
              source='https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-system_cpu_set_information')
(root/'logs/native26_cpu_sets.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(dict(cpu_sets=len(rows), efficiency_classes=classes,
                     single_thread_core_logical_ids=result['single_thread_core_logical_ids'])), flush=True)
