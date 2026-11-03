"""Download/verify the pinned original; never edit the original checkpoint files."""
import hashlib
import json
from runtime import ROOT,REVISION,configure,save


def main():
    configure()
    from huggingface_hub import snapshot_download
    path=ROOT/'models/Ouro-1.4B'
    if not (path/'model.safetensors').exists():
        assert path.resolve().is_relative_to(ROOT.resolve()),'New model download must remain inside V2'
        # configure() protects the evaluation paths with offline defaults. This
        # explicit download command is the only opt-in online preparation path.
        import os
        os.environ['HF_HUB_OFFLINE']='0'
        from huggingface_hub import constants
        constants.HF_HUB_OFFLINE=False
        path.mkdir(parents=True,exist_ok=True)
        snapshot_download(repo_id='ByteDance/Ouro-1.4B',revision=REVISION,local_dir=str(path),
            allow_patterns=['*.json','*.py','*.safetensors','tokenizer*','*.model','README.md','LICENSE*','NOTICE*'])
    h=hashlib.sha256()
    with (path/'model.safetensors').open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    assert h.hexdigest()=='58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af'
    metadata=path/'snapshot_metadata.json'
    if metadata.exists():assert json.loads(metadata.read_text())['sha']==REVISION
    else:
        assert path.resolve().is_relative_to(ROOT.resolve()),'Do not write metadata into a shared original folder'
        metadata.write_text(json.dumps(dict(repo_id='ByteDance/Ouro-1.4B',sha=REVISION,weight_sha256=h.hexdigest()),indent=2),encoding='utf8')
    receipt=ROOT/'logs/MODEL_SERVER_VERIFIED.json'
    if receipt.exists():
        old=json.loads(receipt.read_text());assert old['revision']==REVISION and old['status']=='passed'
    else:save('logs/MODEL_SERVER_VERIFIED.json',dict(status='passed',revision=REVISION,weight_sha256=h.hexdigest(),verification='pinned byte-for-byte original safetensors; strict loading separately enforced at replay'))
    print('PINNED_ORIGINAL_READY',REVISION,h.hexdigest(),flush=True)


if __name__=='__main__':main()
