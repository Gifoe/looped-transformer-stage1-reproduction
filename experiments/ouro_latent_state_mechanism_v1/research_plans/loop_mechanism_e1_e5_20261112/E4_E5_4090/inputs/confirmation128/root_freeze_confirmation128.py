"""Order006: outcome-free selection from existing fixed-token Train inputs."""
from pathlib import Path
import datetime,gzip,hashlib,json

B=Path('D:/ouro-e4-e5-20261112')
OUT=B/'inputs/confirmation128'
PREFIX='ouro-e4-e5-confirm-v1|'


def sha(data):return hashlib.sha256(data).hexdigest()
def write(name,value):
    p=OUT/name;p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');return p


def main():
    if OUT.exists() and any(OUT.iterdir()):raise RuntimeError('Refuse to overwrite a preregistered cohort')
    OUT.mkdir(parents=True,exist_ok=True)
    source=B/'inputs/runner_inputs.jsonl';allrows=[json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    assert len(allrows)==2590 and all('gold' not in r for r in allrows)
    byid={r['id']:r for r in allrows};assert len(byid)==2590
    demos=set(json.loads((B/'inputs/fixed25_demo_ids.json').read_text()));assert len(demos)==25
    ours=set(json.loads((B/'inputs/train64_selection.json').read_text())['ids']);assert len(ours)==64
    peerfile=B/'inputs/peer_e1_discovery64/rows.jsonl.gz'
    with gzip.open(peerfile,'rt',encoding='utf8') as f:peerrows=[json.loads(s) for s in f if s.strip()]
    peers={r['id'] for r in peerrows};assert len(peers)==64 and all(byid[i]['split']=='train' for i in peers)
    usedids=ours|peers|{r['id'] for r in allrows if r['split']=='validation'}
    usedhash={byid[i]['input_ids_sha256'] for i in usedids}
    excluded=[];eligible=[]
    for r in allrows:
        if r['split']!='train':continue
        reasons=[]
        if r['id'] in demos:reasons.append('FIXED_25SHOT_DEMONSTRATION')
        if r['id'] in ours:reasons.append('OUR_OPERATOR_SELECTION_TRAIN64')
        if r['id'] in peers:reasons.append('PEER_OPERATOR_SELECTION_TRAIN64')
        if r['input_ids_sha256'] in usedhash:reasons.append('INPUT_HASH_ALREADY_IN_OPERATOR_SELECTION_COHORTS')
        rank=sha((PREFIX+r['id']).encode('utf8'))
        item={'id':r['id'],'split':'train','selection_hash':rank,'input_ids_sha256':r['input_ids_sha256'],'metadata_sha256':r['metadata_sha256']}
        if reasons:excluded.append({**item,'reasons':reasons})
        else:eligible.append(item)
    eligible.sort(key=lambda r:(r['selection_hash'],r['id']));assert len(eligible)>=128
    selected=eligible[:128];ids=sorted(r['id'] for r in selected);assert len(set(ids))==128
    assert not(set(ids)&(demos|ours|peers)) and all(byid[i]['input_ids_sha256'] not in usedhash for i in ids)
    cohort={'status':'PREREGISTERED_BEFORE_ANY_NEW_128_INTERVENTION','ids':ids,'precision_ids':ids[:8],'technical_ids':ids[:2],
            'selection_rule':'first128 by SHA256(UTF8(ouro-e4-e5-confirm-v1| + ID)), then lexicographic execution/donor order',
            'historical_exposure':'Previous full-cohort descriptions/eight corners and possible peer probe fitting; not globally unseen or blind.',
            'operator_response_scope':'No new causal-operator responses on these questions were used to select this task operators.',
            'no_result_driven_extension':True,'questions':selected}
    write('cohort.json',cohort);write('eligible_hash_ranking.json',eligible);write('excluded_train_ids.json',sorted(excluded,key=lambda r:r['id']))
    write('control_donor_map.json',{i:ids[(n+1)%128] for n,i in enumerate(ids)})
    inputs=[]
    for ident in ids:
        r=byid[ident].copy();r.pop('historical_scores',None);assert 'gold' not in r and 'historical_scores' not in r;inputs.append(r)
    p=OUT/'inputs.jsonl'
    with p.open('w',encoding='utf8',newline='\n') as f:
        for r in inputs:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    # Outcome-bearing labels are joined only AFTER the outcome-free cohort is fixed.
    labels=json.loads((B/'inputs/labels.json').read_text());write('analysis_labels.json',{'train:'+i:labels['train:'+i] for i in ids})
    orders={p.name:sha(p.read_bytes()) for p in (B/'orders').glob('ASTRA_ORDER_*.md')}
    manifest={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'n':128,'eligible_n':len(eligible),'excluded_train_n':len(excluded),
              'selection_script_sha256':sha(Path(__file__).read_bytes()),'input_source_sha256':sha(source.read_bytes()),
              'demo_membership_file_sha256':sha((B/'inputs/fixed25_demo_ids.json').read_bytes()),'peer_cohort_rows_sha256':sha(peerfile.read_bytes()),
              'operator_orders_sha256':orders,'ordered_split_id_sha256_compact_no_newline':sha(json.dumps([['train',i] for i in ids],separators=(',',':')).encode()),
              'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in OUT.iterdir() if p.is_file()}}
    write('FREEZE_RECEIPT.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='files'}))


if __name__=='__main__':main()
