"""Refresh direct-file hashes after visual QA; never alter archived raw records."""
import argparse,hashlib,json
from pathlib import Path
from runtime import ROOT,save


def info(relative):
    p=(ROOT/relative).resolve();assert p.is_relative_to(ROOT.resolve()) and p.is_file()
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return dict(path=relative,bytes=p.stat().st_size,sha256=h.hexdigest())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--add',action='append',default=[]);a=ap.parse_args()
    assert json.loads((ROOT/'results/PACKAGE_COMPLETE.json').read_text())['status']=='complete'
    index=json.loads((ROOT/'ARTIFACT_INDEX.json').read_text())
    public=json.loads((ROOT/'publication/PUBLIC_FILE_INDEX.json').read_text())
    paths={r['path'] for r in public['files']}|set(a.add)|{'scripts/refresh_public_index.py'}
    assert 'ARTIFACT_INDEX.json' in paths
    index['direct_files']=[info(p) for p in sorted(paths) if p!='ARTIFACT_INDEX.json']
    index['post_package_visual_QA']='Only direct-file plot layout/documentation hashes refreshed; archived exact weights and numerical records unchanged.'
    save('ARTIFACT_INDEX.json',index)
    public['files']=[info(p) for p in sorted(paths)];public['total_bytes']=sum(r['bytes'] for r in public['files'])
    save('publication/PUBLIC_FILE_INDEX.json',public)
    print(json.dumps(dict(status='complete',files=len(paths),bytes=public['total_bytes'])))


if __name__=='__main__':main()
