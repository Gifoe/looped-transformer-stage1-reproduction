"""Copy pinned lightweight tokenizers for CPU-only record verification."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from common import ROOT,BASE,save


def main():
    from transformers import AutoTokenizer
    exports=[]
    for model in ['Ouro-1.4B','Ouro-2.6B']:
        source=BASE/'models'/model;target=ROOT/'sources/tokenizers'/model
        target.mkdir(parents=True,exist_ok=True)
        files=[]
        for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json',
                     'merges.txt','vocab.json','config.json','snapshot_metadata.json']:
            old=source/name
            if not old.exists():continue
            new=target/name
            shutil.copyfile(old,new)
            digest=hashlib.sha256(old.read_bytes()).hexdigest()
            assert hashlib.sha256(new.read_bytes()).hexdigest()==digest
            files.append(dict(file=name,bytes=old.stat().st_size,sha256=digest))
        original=AutoTokenizer.from_pretrained(source,local_files_only=True)
        exported=AutoTokenizer.from_pretrained(target,local_files_only=True)
        probes=['Question: 145 + 281 + 62 =','#### -1,234.50\n','中文 × λ\n\nAnswer:',original.eos_token]
        for text in probes:
            a=original.encode(text,add_special_tokens=False)
            b=exported.encode(text,add_special_tokens=False)
            assert a==b and original.decode(a,skip_special_tokens=True)==exported.decode(b,skip_special_tokens=True)
        assert original.eos_token_id==exported.eos_token_id
        exports.append(dict(model=model,revision=json.loads((source/'snapshot_metadata.json').read_text())['sha'],
                            files=files,tokenizer_class=type(exported).__name__,probes_passed=len(probes)))
    save('sources/tokenizer_export_manifest.json',dict(passed=True,exports=exports,scope='Byte-identical tokenizer/config metadata only, no model weight tensors; independent CPU token/text audits'))
    archive=ROOT/'transport/tokenizer_assets.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as out:
        for path in sorted((ROOT/'sources/tokenizers').rglob('*')):
            if path.is_file():out.write(path,path.relative_to(ROOT/'sources/tokenizers'))
    print(json.dumps(dict(passed=True,models=len(exports),archive_bytes=archive.stat().st_size)),flush=True)


if __name__=='__main__':main()
