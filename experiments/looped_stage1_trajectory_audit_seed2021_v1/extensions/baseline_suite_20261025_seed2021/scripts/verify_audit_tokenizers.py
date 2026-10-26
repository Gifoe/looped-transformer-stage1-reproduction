"""CPU portability/source-hash check; optional actual pilot-payload audit."""
import argparse
import hashlib
import json
from common import ROOT,read,save


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot-extra',action='store_true');args=parser.parse_args()
    from transformers import AutoTokenizer
    manifest=json.loads((ROOT/'sources/tokenizer_export_manifest.json').read_text());assert manifest['passed']
    checked=pilot_count=git_blobs=0
    for entry in manifest['exports']:
        model=entry['model'];directory=ROOT/'sources/tokenizers'/model
        loading=json.loads((ROOT/f'results/model_load_{model}.json').read_text())
        assert loading['revision']==entry['revision']
        for file in entry['files']:
            assert '/' not in file['file'] and '\\' not in file['file']
            payload=(directory/file['file']).read_bytes()
            assert len(payload)==file['bytes'] and hashlib.sha256(payload).hexdigest()==file['sha256']
            checked+=1
        if model=='Ouro-2.6B':
            upstream=json.loads((ROOT/'sources/ouro26_hub.json').read_text())
            assert upstream['sha']==entry['revision']
            blobs={r['rfilename']:r.get('blobId') for r in upstream['siblings']}
            for file in entry['files']:
                name=file['file']
                if not blobs.get(name):continue
                payload=(directory/name).read_bytes()
                assert hashlib.sha1(b'blob '+str(len(payload)).encode()+b'\0'+payload).hexdigest()==blobs[name]
                git_blobs+=1
        tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
        assert type(tokenizer).__name__==entry['tokenizer_class']
        assert json.loads((directory/'snapshot_metadata.json').read_text())['sha']==entry['revision']
        assert tokenizer.decode(tokenizer.encode('#### 42\n',add_special_tokens=False),skip_special_tokens=True)=='#### 42\n'
        if args.pilot_extra:
            from audit_suite import check_generated_payload
            rows=read(f'trajectories/math_{model}_pilot_extra.jsonl')
            assert len(rows)==20 and {r['method'] for r in rows}=={'depth_2','depth_3','logits_paper_03','logits_adaptive_paper'}
            for row in rows:check_generated_payload(row,tokenizer,loading['config']['vocab_size'])
            pilot_count+=len(rows)
    result=dict(passed=True,tokenizer_files=checked,pinned_ouro26_git_blobs_verified=git_blobs,
                actual_pilot_payloads=pilot_count,cpu_only=True,
                scope='Exported tokenizer byte hashes/revisions and CPU loading; optional true-depth/fixed/adaptive pilot token/text/EM/trace integrity. Does not establish formal test completion.')
    save('results/tokenizer_portability_'+('server_pilot' if args.pilot_extra else 'local_cpu')+'.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
