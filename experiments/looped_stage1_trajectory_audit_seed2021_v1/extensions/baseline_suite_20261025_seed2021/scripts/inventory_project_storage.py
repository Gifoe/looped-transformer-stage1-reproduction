"""Read-only apparent-byte inventory of explicitly named task-owned folders."""
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
base = root.parents[1]
assert str(base).replace('\\', '/').lower() == 'd:/looped_stage1_trajectory_audit_seed2021_v1'
owned = [base, Path('D:/loopbase_env_20261025'), Path('E:/loopbase_20261025_seed2021')]
categories = []

def scan(path):
    files = size = directories = 0
    skipped = []
    def fail(error):
        raise error
    for current, children, names in os.walk(path, topdown=True, followlinks=False, onerror=fail):
        directories += 1
        for name in list(children):
            child = Path(current)/name
            if child.is_symlink() or child.is_junction():
                skipped.append(str(child))
                children.remove(name)
        for name in names:
            child = Path(current)/name
            if child.is_symlink():
                skipped.append(str(child))
                continue
            size += child.stat().st_size
            files += 1
    return dict(path=str(path), files=files, directories=directories,
                apparent_bytes=size, skipped_reparse_paths=skipped)

for path in owned:
    assert path.is_dir()
    categories.append(scan(path))

class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                ('total_physical', ctypes.c_ulonglong), ('available_physical', ctypes.c_ulonglong),
                ('total_pagefile', ctypes.c_ulonglong), ('available_pagefile', ctypes.c_ulonglong),
                ('total_virtual', ctypes.c_ulonglong), ('available_virtual', ctypes.c_ulonglong),
                ('available_extended_virtual', ctypes.c_ulonglong)]

memory = MemoryStatus()
memory.length = ctypes.sizeof(memory)
assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
total = sum(item['apparent_bytes'] for item in categories)
result = dict(passed=True, checked_utc=datetime.now(timezone.utc).isoformat(),
              folders=categories, task_owned_apparent_bytes=total,
              within_preferred_60GB=(total < 60_000_000_000),
              memory=dict(total_physical_bytes=memory.total_physical,
                          available_physical_bytes=memory.available_physical,
                          load_percent=memory.load),
              free_bytes={drive: shutil.disk_usage(drive+':/').free for drive in ['D', 'E', 'F']},
              scope='Apparent file bytes, including task caches, failed env and transport snapshots; junctions not traversed or double-counted. Existing shared Anaconda is not a task-owned allocation. No files moved/deleted, no model bytes read.')
(root/'logs/storage_inventory_20261026.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result), flush=True)
