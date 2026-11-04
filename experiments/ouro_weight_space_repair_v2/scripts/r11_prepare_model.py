"""Verify pinned model bytes and optionally create missing runtime metadata.

Never downloads or changes weights, tokenizer or code. Existing different files
are refused. External model symlinks remain read-only.
"""
import argparse,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main(prepare):
    identity=json.loads((ROOT/'configs/r11_PROTOCOL.json').read_text(encoding='utf8'))['model_identity'];model=ROOT/'models/Ouro-1.4B'
    assert model.is_dir() and sha(model/'model.safetensors')==identity['safetensors_sha256']
    metadata=identity['metadata_and_tokenizer_sha256']
    for name,expected in metadata.items():
        if name=='snapshot_metadata.json':continue
        assert sha(model/name)==expected,('Pinned metadata/tokenizer/source differs',name)
    snapshot=model/'snapshot_metadata.json';created=False
    if not snapshot.exists():
        assert prepare,'Missing custom snapshot metadata: restore released metadata with --prepare after verifying weights'
        assert model.resolve().is_relative_to(ROOT.resolve()),'Do not write through an external reused model symlink'
        raw=(ROOT/'artifacts/r11_MODEL_SNAPSHOT_METADATA.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==metadata['snapshot_metadata.json']
        with snapshot.open('xb') as f:f.write(raw)
        created=True
    assert sha(snapshot)==metadata['snapshot_metadata.json'] and json.loads(snapshot.read_text(encoding='utf8'))['sha']==identity['revision']
    receipt=ROOT/'logs/MODEL_SERVER_VERIFIED.json'
    if receipt.exists():
        value=json.loads(receipt.read_text(encoding='utf8'));assert value['status']=='passed' and value['revision']==identity['revision']
    elif prepare:
        assert receipt.parent.is_dir();value=dict(status='passed',revision=identity['revision'],safetensors_sha256=identity['safetensors_sha256'],metadata_and_tokenizer_sha256=metadata,
            source='Independent local byte verification against frozenR11 identity; no weights/code were modified')
        with receipt.open('x',encoding='utf8') as f:f.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='verified_model_bytes',runtime_ready=receipt.exists(),snapshot_metadata_created=created,prepare=prepare,revision=identity['revision'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args();main(a.prepare)
