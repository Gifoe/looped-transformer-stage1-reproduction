"""Metadata-only index of a user-provided large data volume (e.g., 6.1 TB).

Never opens file content, never modifies files. Does not equate storage to dataset size.
"""
from __future__ import annotations
import argparse
import heapq
import json
from collections import Counter
from pathlib import Path
from core import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("analyses/volume_inventory.json"))
    a=p.parse_args()
    root=a.root.resolve()
    if not root.is_dir():raise NotADirectoryError(root)
    files=0;size=0;by_ext=Counter();by_ext_size=Counter();top=[];errors=[]
    stack=[root]
    while stack:
        current=stack.pop()
        try:
            for x in current.iterdir():
                try:
                    if x.is_symlink():continue
                    if x.is_dir():stack.append(x);continue
                    if not x.is_file():continue
                    n=x.stat().st_size;files+=1;size+=n
                    ext=x.suffix.lower() or "<none>"
                    by_ext[ext]+=1;by_ext_size[ext]+=n
                    item=(n,str(x.relative_to(root)))
                    if len(top)<30:heapq.heappush(top,item)
                    elif item>top[0]:heapq.heapreplace(top,item)
                except (PermissionError,OSError) as e: errors.append(f"{x}: {e}")
        except (PermissionError,OSError) as e: errors.append(f"{current}: {e}")
    report=dict(root=str(root),files=files,bytes=size,tib=size/(1024**4),
                extensions=[dict(ext=e,files=by_ext[e],bytes=by_ext_size[e]) for e in by_ext_size.most_common(50)],
                largest_files=[dict(path=f,bytes=n) for n,f in sorted(top,reverse=True)],
                scan_errors=errors[:50],errors_total=len(errors),
                note="Metadata-only inventory; relevance, license and scientific contents need separate inspection")
    write_json(a.out,report)
    print(json.dumps({"files":files,"total_tib":report["tib"],"largest_files":report["largest_files"][:5],"output":str(a.out)},indent=2))

if __name__=="__main__":main()
