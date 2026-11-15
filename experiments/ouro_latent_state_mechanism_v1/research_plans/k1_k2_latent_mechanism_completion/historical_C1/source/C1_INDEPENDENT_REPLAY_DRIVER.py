"""Independent C1 explicit-forward replay. No gold is loaded or used."""
import argparse
import datetime
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda:f.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    os.replace(tmp, path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('/root/autodl-tmp/ouro-e4-e5/model'))
    p.add_argument('--selection', choices=['technical', 'replay'], required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    inputfile = root/'inputs/train.jsonl'
    discoveryfile = Path('/root/autodl-tmp/ouro-e1-e3-offload-20261113/inputs/E1_DISCOVERY64_FROZEN.json')
    rows = [json.loads(line) for line in inputfile.read_text().splitlines()]
    byid = {r['id']:r for r in rows}
    assert len(rows) == len(byid) == 1094
    original = json.loads(discoveryfile.read_text())
    discovery = sorted(r['id'] for group in original['groups'].values() for r in group)
    assert len(discovery) == len(set(discovery)) == 64 and set(discovery) <= set(byid)
    technical = discovery[:2]
    hash8 = sorted(byid, key=lambda x:hashlib.sha256(('C1-replay-v1|'+x).encode()).hexdigest())[:8]
    selected = technical if a.selection == 'technical' else list(dict.fromkeys(technical+hash8))
    source = {str(Path(__file__).name):sha(__file__),
              'C1_AUDIT_TENSOR_IO.py':sha(Path(__file__).with_name('C1_AUDIT_TENSOR_IO.py')),
              'C1_INDEPENDENT_LAYER_REPLAY.py':sha(Path(__file__).with_name('C1_INDEPENDENT_LAYER_REPLAY.py')),
              'train_inputs':sha(inputfile), 'original_discovery':sha(discoveryfile),
              'scientific_order':sha(root/'ASTRA_ORDER_C1_UPSTREAM_TO_L18.md')}
    fixture = {'status':'FROZEN_BEFORE_REPLAY','technical':technical,'hash_first8':hash8,
               'selected':selected,'selection':a.selection,'source_hashes':source,
               'input_rows':{x:hashlib.sha256(json.dumps(byid[x],sort_keys=True,separators=(',',':')).encode()).hexdigest() for x in selected},
               'no_gold':'No labels or outcomes used for these identities.'}
    fixturefile = root/'independent_replay'/('SELECTION_'+a.selection+'.json')
    if fixturefile.exists():
        assert json.loads(fixturefile.read_text()) == fixture
    else:
        write(fixturefile, fixture)
    if a.prepare_only:
        print(json.dumps(fixture,indent=2));return
    assert a.out is not None and not a.out.exists(), 'Use a new immutable replay namespace'
    complete = Path('/root/autodl-tmp/ouro-e7/reencoding/full_capture/COMPLETE.json')
    assert complete.exists(), 'E7 must finish before C1 GPU replay'
    # The process-wide GPU lock is shared with the existing E7/C1 execution queue.
    lock = open('/root/autodl-tmp/ouro-e6a/gpu.lock','a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    import torch
    import transformers
    from safetensors.torch import load_file,save_file
    from C1_AUDIT_TENSOR_IO import load_checked
    from C1_INDEPENDENT_LAYER_REPLAY import independent_forward,tensor_hash,MODES
    assert torch.__version__ == '2.7.1+cu128' and transformers.__version__ == '4.55.0'
    transformers.utils.import_utils._torchvision_available=False
    torch.set_num_threads(1)
    torch.manual_seed(2021)
    torch.backends.cuda.matmul.allow_tf32=False
    expected = {'model.safetensors':'58872a72616c736595b8b7662079c5b12c5a162ec16eae94f21c348dfa9885af',
                'modeling_ouro.py':'c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413',
                'config.json':'ce9cc13da41591b8b4deca053d7dfee06424c0228628ee862ea86d725bc163f3',
                'configuration_ouro.py':'950443e32929047aa08d02abad2e1888bc1914b3db988d3d675f70787f65dafb'}
    for filename,digest in expected.items():
        assert sha(a.model/filename) == digest, filename
    start = time.time()
    model = transformers.AutoModelForCausalLM.from_pretrained(str(a.model),trust_remote_code=True,
              torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).eval().to('cuda')
    assert sha(inspect.getfile(type(model))) == expected['modeling_ouro.py']
    assert model.config._attn_implementation == 'sdpa'
    assert '5090' in torch.cuda.get_device_name()
    a.out.mkdir(parents=True)
    write(a.out/'STARTED.json', {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'selection_sha256':sha(fixturefile),'source_hashes':source,'model_hashes':expected,
          'E7_complete_sha256':sha(complete),'pid':os.getpid(),'argv':sys.argv,
          'gpu':torch.cuda.get_device_name(),'model_load_seconds':time.time()-start})
    cases = []
    expected_mode = {'NXU':'UA','NAU':'UX','NXUAU':'U'}
    for ident in selected:
        row = byid[ident]
        stem = hashlib.sha256(ident.encode()).hexdigest()
        refjson = root/'references/train'/(stem+'.json')
        refst = refjson.with_suffix('.safetensors')
        refmeta = json.loads(refjson.read_text())
        tensors,refarchive = load_checked(refst,refmeta)
        deltas = {i:tensors[f'd_{i:02d}'] for i in range(9,16)}
        branches = {key:tensors[tkey] for key,tkey in [('xN','N_x_R'),('aN','N_a_R'),('xU','U_x_R'),('aU','U_a_R')]}
        ids = torch.tensor([row['ids']],device='cuda',dtype=torch.long)
        # Original Joint input manifests hash the JSON integer list, not tensor bytes.
        assert hashlib.sha256(json.dumps(row['ids']).encode()).hexdigest() == row['input_ids_sha256']
        case = {'id':ident,'reference_json_sha256':sha(refjson),'reference_tensor_sha256':refmeta['tensor_sha256'],
                'reference_archive':refarchive,'modes':{}}
        saved = {}
        for mode in MODES:
            match = expected_mode.get(mode, mode)
            if match in ('N','P','U'):
                compare = refmeta['conditions'][match]
                endpoint = tensors[match+'_logits']
                origin = {'reference_condition':match}
            else:
                path = root/'treatments/train'/stem/(match+'.json')
                compare = json.loads(path.read_text())
                tensorfile = path.with_suffix('.safetensors')
                content,archive=load_checked(tensorfile,compare)
                endpoint = content['logits']
                origin = {'treatment_json_sha256':sha(path),'treatment_tensor_sha256':compare['tensor_sha256'],
                          'treatment_archive':archive,'condition':match}
            replay = independent_forward(model,ids,row['R'],row['prefix_len'],row['targets'],deltas,branches,mode)
            assert replay['full_vocab_R_sha256'] == compare['Rlogits_sha256'], (ident,mode,'hash')
            assert tuple(endpoint.shape) in [(49152,), (1,49152)]
            assert torch.equal(replay['logits'],endpoint.reshape(1,49152)), (ident,mode,'endpoint')
            assert replay['candidate_logits'] == compare['candidate_logits'], (ident,mode,'candidates')
            assert replay['scores'] == compare['scores'], (ident,mode,'scores')
            item = {k:v for k,v in replay.items() if k not in ('logits','tensors')}
            item.update(origin)
            for i,entry in replay['tensors'].items():
                for key,value in entry.items():
                    if isinstance(value,torch.Tensor):
                        saved[f'{mode}_L{i+1:02d}_{key}'] = value.contiguous()
            saved[mode+'_logits'] = replay['logits'].contiguous()
            case['modes'][mode] = item
            print('C1_INDEPENDENT_PASS',ident,mode,flush=True)
        outtensor = a.out/(stem+'.safetensors')
        save_file(saved,str(outtensor))
        case['tensor_sha256'] = sha(outtensor)
        write(a.out/(stem+'.json'),case)
        cases.append({'id':ident,'metadata_sha256':sha(a.out/(stem+'.json')),'tensor_sha256':sha(outtensor)})
    for filename,digest in expected.items():
        assert sha(a.model/filename) == digest
    assert source['C1_INDEPENDENT_LAYER_REPLAY.py'] == sha(Path(__file__).with_name('C1_INDEPENDENT_LAYER_REPLAY.py'))
    write(a.out/'COMPLETE.json',{'status':'PASS','selection_sha256':sha(fixturefile),'source_hashes':source,
          'cases':cases,'wall_seconds':time.time()-start,'scope':'Independent explicit-layer intervention and complete BF16 endpoints, not independent questions or model seeds.'})
    print('C1_INDEPENDENT_COMPLETE',len(cases),flush=True)

if __name__ == '__main__':main()
