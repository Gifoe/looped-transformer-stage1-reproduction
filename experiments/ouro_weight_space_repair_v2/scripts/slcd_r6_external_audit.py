"""Independent CPU-only audit of COMPLETED frozen R6 external evaluations.

Default audits both tasks. Even --task requires both COMPLETE files and the
chain COMPLETE before any outcome file is read. --self-test uses synthetic data
only. Never imports the external worker, model runtime, torch, or all_scores.
"""
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
PRIMARY='r6_prior_projection'
METHODS=('native4','fixed','adaptive','A1_g1p0',PRIMARY)
BASE={f'native_k{k}' for k in range(1,7)}|{'loopcd_k4_fixed','loopcd_k4_adaptive'}
SIZES={'hellaswag':10042,'openbookqa':500}
LOCK='configs/slcd_R6_EXTERNAL_LOCK.json'


def require(condition,message):
    if not condition:raise ValueError(message)


def path(relative):
    value=(ROOT/relative).resolve()
    require(value.is_relative_to(ROOT.resolve()),'Artifact outside root: '+str(relative))
    return value


def read_path(relative):
    """Read pinned model assets through the existing read-only V1 symlink.

    This exception never applies to output paths. Each model asset is separately
    bound to the frozen expected byte SHA before its contents are used.
    """
    parts=Path(relative).parts
    if parts[:2]==('models','Ouro-1.4B'):
        allowed={'model.safetensors','config.json','configuration_ouro.py','merges.txt',
                 'modeling_ouro.py','snapshot_metadata.json','special_tokens_map.json',
                 'tokenizer.json','tokenizer_config.json','vocab.json'}
        require(len(parts)==3 and parts[2] in allowed,'Unexpected model asset path')
        model_root=(ROOT/'models/Ouro-1.4B').resolve()
        value=(ROOT/relative).resolve()
        require(value.is_relative_to(model_root) and value.is_file(),'Pinned model asset outside model root')
        return value
    return path(relative)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def file_sha(value):
    h=hashlib.sha256()
    with Path(value).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(relative):return json.loads(path(relative).read_text(encoding='utf-8'))


def jsonlines(relative):
    with path(relative).open(encoding='utf-8') as stream:
        for number,line in enumerate(stream,1):
            if line.strip():
                value=json.loads(line)
                require(isinstance(value,dict),f'Non-object row {relative}:{number}')
                yield value


def read_index(relative,key='id'):
    result={}
    for row in jsonlines(relative):
        require(row[key] not in result,'Duplicate key in '+relative);result[row[key]]=row
    return result


def finite_vector(value,n,label):
    raw=np.asarray(value)
    require(raw.ndim==1 and raw.size==n and raw.dtype.kind in 'iuf','Invalid score shape/type: '+label)
    result=raw.astype(np.float64)
    require(np.isfinite(result).all(),'Nonfinite score: '+label)
    return result


def check_readouts(methods,lengths,gold):
    require(set(methods)==BASE,'Native readout method set differs')
    denominator=np.asarray(lengths,dtype=np.float64)
    require(np.isfinite(denominator).all() and (denominator>0).all(),'Invalid character lengths')
    for name,value in methods.items():
        raw=finite_vector(value['sum_scores'],len(lengths),name)
        normalized=raw/denominator
        require(np.array_equal(normalized,finite_vector(value['normalized_scores'],len(lengths),name)),name+' double/wrong normalization')
        prediction=int(normalized.argmax())
        require(value['prediction']==prediction and value['correct']==(prediction==gold),name+' prediction/correctness differs')


def ranking_scores(original,prior,selected):
    """Independently derive only the five frozen outputs and cross-check kernel."""
    f=np.asarray(original['loopcd_k4_fixed']['normalized_scores'],dtype=np.float64)
    k4=np.asarray(original['native_k4']['normalized_scores'],dtype=np.float64)
    k5=np.asarray(original['native_k5']['normalized_scores'],dtype=np.float64)
    adaptive=np.asarray(original['loopcd_k4_adaptive']['normalized_scores'],dtype=np.float64)
    e=np.asarray(prior['loopcd_k4_fixed']['normalized_scores'],dtype=np.float64)-np.asarray(prior['native_k4']['normalized_scores'],dtype=np.float64)
    g=f-k4;gc=g-g.mean();ec=e-e.mean();den=float(ec@ec)
    alpha=0. if den==0 else float(np.clip(float(gc@ec)/den,0.,1.))
    a1=f+(k4-k5);independent=a1-alpha*e
    actual=selected(original,prior,PRIMARY)
    require(np.allclose(independent,actual,rtol=0,atol=1e-12),'Independent formula differs from frozen selected kernel')
    return {'native4':k4,'fixed':f,'adaptive':adaptive,'A1_g1p0':a1,PRIMARY:actual}


def statistics(good,old,rng):
    """Independent paired-category counting, exact McNemar and seeded bootstrap."""
    from scipy.stats import binomtest
    good=np.asarray(good,dtype=bool);old=np.asarray(old,dtype=bool)
    require(good.shape==old.shape and good.ndim==1 and good.size>0,'Invalid paired arrays')
    n=good.size;rescue=int(np.count_nonzero(good&~old));damage=int(np.count_nonzero(old&~good))
    sample=rng.multinomial(n,np.array([rescue,damage,n-rescue-damage],dtype=float)/n,size=50000)
    distribution=100*(sample[:,0]-sample[:,1])/n
    lo,hi=[float(x) for x in np.quantile(distribution,[.025,.975])]
    return dict(n=int(n),correct=int(good.sum()),accuracy=float(good.mean()),corrections=rescue,regressions=damage,
        net_correct=rescue-damage,delta_accuracy_pp=100*(rescue-damage)/n,delta_pp=100*(rescue-damage)/n,
        exact_mcnemar_p=1. if rescue+damage==0 else float(binomtest(rescue,rescue+damage,.5).pvalue),
        paired_bootstrap_95ci_low_pp=lo,paired_bootstrap_95ci_high_pp=hi,
        paired_bootstrap_low_pp=lo,paired_bootstrap_high_pp=hi,bootstrap_replicates=50000)


def equal_fields(actual,expected,label,csv_row=False):
    for key,value in expected.items():
        require(key in actual,label+' missing '+key)
        x=actual[key]
        if csv_row:
            if isinstance(value,bool):require(x in ('True','False'),label+' malformed Boolean');x=x=='True'
            elif isinstance(value,int):x=int(x)
            elif isinstance(value,float):x=float(x)
        if isinstance(value,float):require(math.isfinite(float(x)) and abs(float(x)-value)<=1e-12,label+' differs: '+key)
        else:require(x==value,label+' differs: '+key)


class Audit:
    def __init__(self):self.hashes={};self.dependency_hashes={};self.token_pairs_checked=0;self.new_used=set()

    def bind(self,relative,expected=None):
        value=file_sha(read_path(relative))
        if expected is not None:require(value==expected,'SHA differs: '+relative)
        self.hashes[relative]=value;return value

    def preconditions(self):
        # No outcomes, caches or manifests may be opened before these checks.
        required=['results/slcd_R6_EXTERNAL_CHAIN_COMPLETE.json']+[f'results/slcd_r6_{t}_COMPLETE.json' for t in SIZES]
        require(all(path(p).is_file() for p in required),'Both task COMPLETE files and chain COMPLETE are required; do not audit partial runs')
        chain=read(required[0]);require(chain['status']=='complete' and chain['method_unchanged'] is True,'External chain incomplete')
        for task,n in SIZES.items():
            r=read(f'results/slcd_r6_{task}_COMPLETE.json')
            require(r['status']=='complete' and r['n']==n and chain['tasks'][task]['status']=='complete','Task incomplete: '+task)
        self.chain=chain

    def initialize(self):
        self.preconditions()
        self.lock=read(LOCK);self.lock_sha=self.bind(LOCK);self.final=read('configs/slcd_R6_FINAL_LOCK.json')
        lock=self.lock;final=self.final
        self.final_sha=self.bind('configs/slcd_R6_FINAL_LOCK.json',lock['expected_final_sha'])
        require(lock['primary']==final['primary']==PRIMARY and lock['num_fewshot']==0 and lock['metric']=='acc_norm','External method/protocol differs')
        require(self.chain['external_lock_sha256']==self.lock_sha,'Chain lock binding differs')
        self.bind('results/slcd_R6_EXTERNAL_CHAIN_COMPLETE.json')
        self.bind('scripts/slcd_r6_external_worker.py',lock['worker_source_sha256'])
        self.bind('scripts/slcd_r6_scores.py',lock['ranking_source_sha256'])
        require(lock['ranking_source_sha256']==final['ranking_source_sha256'],'Final ranking hash differs')
        for name,h in lock['inference_sources'].items():self.bind('scripts/'+name,h)
        for key,name in [('controller_sha256','slcd_r6_external_chain.py'),('preparation_source_sha256','slcd_r6_prepare_external.py')]:self.bind('scripts/'+name,lock[key])
        self.bind('configs/slcd_R6_EXTERNAL_DATA.json',lock['external_data_sha256'])
        require(read('configs/slcd_R6_EXTERNAL_DATA.json')['tasks']==lock['tasks'],'Preparation/task definitions differ')
        self.bind('configs/slcd_R6_DEVELOPMENT_LOCK.json',final['development_lock_sha256'])
        dev=read('configs/slcd_R6_DEVELOPMENT_LOCK.json')
        self.bind('scripts/slcd_r5_evaluate.py',dev['imported_prior_evaluator_sha256'])
        self.bind('configs/slcd_R6_METHODS.json',final['methods_config_sha256'])
        self.bind('scripts/slcd_r6_evaluate.py',final['evaluation_source_sha256'])
        self.bind('scripts/slcd_r2_evaluate.py',final['imported_evaluation_source_sha256'])
        self.bind('results/slcd_R6_TEST_ACCEPTANCE.json',lock['arc_acceptance_sha256'])
        self.bind('results/slcd_R6_ARC_COMPLETE.json',lock['arc_result_sha256'])
        acceptance=read('results/slcd_R6_TEST_ACCEPTANCE.json');gate=read('results/slcd_R6_ARC_COMPLETE.json')
        require(acceptance['status']=='passed' and gate['status']=='complete' and acceptance['n']==gate['n']==1172
                and acceptance['correct']==gate['correct']>=733 and acceptance['primary']==gate['primary']==PRIMARY
                and acceptance['final_lock_sha256']==gate['final_lock_sha256']==self.final_sha,'ARC authorization gate differs')
        self.versions={name:importlib.metadata.version(name) for name in lock['package_versions']}
        require(self.versions==lock['package_versions'],'Audit environment differs from locked environment; use server CPU env')
        require(lock['model_identity']==final['model_identity'],'Model identities differ')
        identity=final['model_identity'];self.bind('models/Ouro-1.4B/model.safetensors',identity['safetensors_sha256'])
        for name,h in identity['metadata_and_tokenizer_sha256'].items():self.bind('models/Ouro-1.4B/'+name,h)
        # Load the tokenizer only. Extract the tiny official pair function without
        # importing lm_eval/transformers/torch or executing its surrounding module.
        from tokenizers import Tokenizer
        self.tokenizer=Tokenizer.from_file(str(read_path('models/Ouro-1.4B/tokenizer.json')))
        self.harness_distribution=importlib.metadata.distribution('lm_eval')
        source_path=self.harness_distribution.locate_file('lm_eval/api/model.py')
        source=Path(source_path).read_text(encoding='utf-8');tree=ast.parse(source)
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='TemplateLM')
        fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_encode_pair')
        segment='\n'.join(source.splitlines()[fn.lineno-1:fn.end_lineno])+'\n'
        self.pair_source_sha=hashlib.sha256(segment.encode()).hexdigest()
        source_audit=read('results/slcd_SOURCE_AUDIT.json');self.bind('results/slcd_SOURCE_AUDIT.json')
        require(self.pair_source_sha==source_audit['encode_pair_source_sha256'],'Official pair source differs from audited inference source')
        namespace={'List':list,'Tuple':tuple,'transformers':SimpleNamespace(AutoModelForSeq2SeqLM=object())}
        exec(textwrap.dedent(segment),namespace)
        self.pair_function=namespace['_encode_pair']
        self.coder=SimpleNamespace(tok_encode=lambda value,**kw:self.tokenizer.encode(value,add_special_tokens=False).ids)
        sys.path.insert(0,str(path('scripts')))
        from slcd_r6_scores import selected_score
        self.selected=selected_score
        self.initialize_caches(acceptance)

    def pair(self,context,continuation):
        answer=self.pair_function(self.coder,context,continuation)
        require(answer[0] and answer[1],'Empty encoded prefix/target')
        self.token_pairs_checked+=1;return answer

    def cache_entry(self,entry,signature):
        require(entry['signature']==signature and entry['prefix']=='Answer:' and set(entry['sum_scores'])==BASE,'Prior cache signature/method set differs')
        require(all(math.isfinite(float(v)) for v in entry['sum_scores'].values()),'Nonfinite cache sum')
        require(entry['key']==digest(dict(global_signature=signature,prefix='Answer:',continuation=entry['continuation'],
            context_ids=entry['context_ids'],target_ids=entry['target_ids'])),'Prior cache key differs')
        require(self.pair('Answer:',entry['continuation'])==(entry['context_ids'],entry['target_ids']),'Prior cache token IDs differ')

    def initialize_caches(self,acceptance):
        old_receipt_path='results/slcd_r6_priors_test_COMPLETE.json'
        self.bind(old_receipt_path,acceptance['prior_receipt_sha256']);old=read(old_receipt_path);prov=old['provenance']
        require(old['status']=='complete' and old['n']==1172 and prov['authorization_lock_sha256']==self.final_sha,'ARC prior authority incomplete')
        require(prov['model_identity']==self.final['model_identity'],'ARC prior model identity differs')
        require(all(importlib.metadata.version(k)==v for k,v in prov['package_versions'].items()),'ARC prior environment differs')
        for name,h in prov['inference_source_sha256'].items():self.bind('scripts/'+name,h)
        self.old_signature=digest(dict(model_identity=prov['model_identity'],implementation=prov['inference_source_sha256'],
            package_versions=prov['package_versions'],prefix='Answer:',seed=2021,scoring='full target LL raw sums; original BF16 and FP32 logsoftmax'))
        require(self.old_signature==prov['global_cache_signature'],'ARC prior global signature differs')
        self.bind(old['score_file'],old['score_file_sha256']);approved={};ids=set()
        for row in jsonlines(old['score_file']):
            require(row['id'] not in ids and row['signature']==old['signature'],'ARC prior duplicate/signature differs');ids.add(row['id'])
            require(len(row['cache_keys'])==len(row['cache_entry_sha256']),'ARC prior index lengths differ')
            for key,value in zip(row['cache_keys'],row['cache_entry_sha256']):
                require(key not in approved or approved[key]==value,'Conflicting ARC prior entry digests');approved[key]=value
        require(len(ids)==1172,'ARC prior authority coverage differs')
        self.bind('results/slcd_r6_prior_cache.jsonl');self.old_cache=read_index('results/slcd_r6_prior_cache.jsonl','key')
        require(set(self.old_cache)==set(approved),'ARC cache/receipt key coverage differs')
        for key,entry in self.old_cache.items():
            self.cache_entry(entry,self.old_signature);require(digest(entry)==approved[key],'ARC cache entry differs from completed receipt')
        self.raw_signature=digest(dict(external_lock_sha256=self.lock_sha,model_identity=self.final['model_identity'],
            worker_source_sha256=self.lock['worker_source_sha256'],inference_sources=self.lock['inference_sources'],
            package_versions=self.versions,prefix='Answer:',seed=2021,
            scoring='original scorer six-loop raw complete-target LL; no character normalization in cache'))
        self.bind('results/slcd_r6_external_prior_cache.jsonl')
        self.new_cache=read_index('results/slcd_r6_external_prior_cache.jsonl','key')
        for entry in self.new_cache.values():
            self.cache_entry(entry,self.raw_signature)
            require(entry['origin']=='fresh_external_original_model','New cache falsely claims old provider origin')
        self.old_receipt=old

    def task(self,task):
        spec=self.lock['tasks'][task];n=SIZES[task];stem='results/slcd_r6_'+task
        complete_path=stem+'_COMPLETE.json';self.bind(complete_path,self.chain['tasks'][task]['completion_sha256'])
        complete=read(complete_path)
        require(complete['n']==n and complete['task']==task and complete['primary']==PRIMARY,'Task completion header differs')
        require(complete['external_lock_sha256']==self.lock_sha and complete['final_lock_sha256']==self.final_sha,'Task final/lock binding differs')
        require(complete['signature']==digest(complete['provenance']),'Task provenance signature differs')
        prov=complete['provenance']
        require(prov['external_lock_sha256']==self.lock_sha and prov['final_lock_sha256']==self.final_sha
                and prov['manifest_sha256']==spec['manifest_sha256'] and prov['model_identity']==self.final['model_identity']
                and prov['package_versions']==self.versions and prov['global_external_prior_signature']==self.raw_signature,'Task provenance differs')
        require(prov['acceptance_sha256']==self.hashes['results/slcd_R6_TEST_ACCEPTANCE.json']
                and prov['arc_gate_sha256']==self.hashes['results/slcd_R6_ARC_COMPLETE.json'],'Task ARC authority differs')
        equal_fields(prov['old_prior_authority'],dict(receipt_sha256=self.hashes['results/slcd_r6_priors_test_COMPLETE.json'],
            score_sha256=self.old_receipt['score_file_sha256'],cache_sha256=self.hashes['results/slcd_r6_prior_cache.jsonl'],
            global_signature=self.old_signature),'Task old cache authority')
        self.bind(spec['manifest_path'],spec['manifest_sha256']);manifest=read_index(spec['manifest_path'])
        require(len(manifest)==spec['n']==n,'Manifest size differs')
        split='validation' if task=='hellaswag' else 'test'
        require(spec['split']==split and spec['num_fewshot']==0 and spec['seed']==2021,'Official task split/few-shot setting differs')
        require(list(manifest)==[f'{task}/{split}/{i}' for i in range(n)],'Official ordered document IDs differ')
        source_path=f'data/slcd_{task}_raw_SOURCE.json';self.bind(source_path)
        require(read(source_path)==spec['source'],'Raw dataset source receipt differs')
        self.bind(f'data/slcd_{task}_raw.parquet',spec['source']['sha256'])
        for filename,expected_hash in [(task+'.yaml',spec['official_yaml_sha256']),('utils.py',spec['official_utils_sha256'])]:
            if expected_hash is not None:
                relative='lm_eval/tasks/'+task+'/'+filename
                require(file_sha(self.harness_distribution.locate_file(relative))==expected_hash,'Official task source differs: '+relative)
                self.dependency_hashes[relative]=expected_hash
        require(complete['score_file']==stem+'_scores.jsonl','Unexpected score file name')
        self.bind(complete['score_file'],complete['score_file_sha256'])
        require(self.chain['tasks'][task]['score_file_sha256']==complete['score_file_sha256'],'Chain score binding differs')
        old_main=self.old_main(task,spec,manifest,prov)
        compact={};old_count=old_prior_hits=boundary_changes=0;new_used=set()
        for row in jsonlines(complete['score_file']):
            qid=row['id'];require(qid in manifest and qid not in compact,'Unexpected/duplicate scored ID')
            m=manifest[qid]
            require(row['signature']==complete['signature'] and row['primary']==PRIMARY,'Row signature/primary differs')
            for key in ('gold','prompt','prompt_sha256','request_pairs','choice_char_lengths'):require(row[key]==m[key],'Manifest/row mismatch: '+key)
            require(hashlib.sha256(m['prompt'].encode()).hexdigest()==m['prompt_sha256'],'Prompt hash differs')
            require(len(m['request_pairs'])==len(m['choice_char_lengths'])==4 and 0<=m['gold']<4,'Wrong choice count/gold')
            require(all(p[0]==m['prompt'] for p in m['request_pairs']),'Candidate prompt differs')
            pairs=[self.pair(*p) for p in m['request_pairs']];prior_pairs=[self.pair('Answer:',p[1]) for p in m['request_pairs']]
            require(row['choice_token_counts']==[len(t) for _,t in pairs] and row['prior_token_counts']==[len(t) for _,t in prior_pairs],'Target counts differ')
            equality=[t==pt for (_,t),(_,pt) in zip(pairs,prior_pairs)]
            require(row['target_ids_equal_by_choice']==equality,'Target boundary diagnostic differs')
            boundary_changes+=sum(not x for x in equality)
            require(row['weights_modified'] is False and row['actual_shared_depth']==6,'Row execution metadata differs')
            if row['main_origin']=='old_authority':
                require(qid in old_main and row['original_methods']==old_main[qid]['original_methods'],'Old main scores differ');old_count+=1
            else:require(row['main_origin']=='fresh' and qid not in old_main,'Incorrect main origin')
            check_readouts(row['original_methods'],m['choice_char_lengths'],m['gold'])
            check_readouts(row['prior_methods'],m['choice_char_lengths'],m['gold'])
            require(len(row['prior_cache_entries'])==4,'Wrong prior reference count')
            for i,ref in enumerate(row['prior_cache_entries']):
                if ref['origin']=='old_arc_prior_authority':entry=self.old_cache[ref['cache_key']];old_prior_hits+=1
                else:
                    require(ref['origin']=='fresh_external_original_model','Unknown prior origin')
                    entry=self.new_cache[ref['cache_key']];new_used.add(ref['cache_key'])
                require(digest(entry)==ref['entry_sha256'],'Prior entry digest differs')
                context,target=prior_pairs[i];continuation=m['request_pairs'][i][1]
                require(entry['context_ids']==context and entry['target_ids']==target and entry['continuation']==continuation,'Prior candidate input differs')
                require(ref['input_key']==digest(dict(prefix='Answer:',continuation=continuation,context_ids=context,target_ids=target)),'Prior input-key binding differs')
                for name in BASE:require(row['prior_methods'][name]['sum_scores'][i]==entry['sum_scores'][name],'Prior raw LL differs from cache')
            outputs=ranking_scores(row['original_methods'],row['prior_methods'],self.selected)
            require(set(row['methods'])==set(METHODS),'Final ranking method set differs')
            for name,value in outputs.items():
                actual=row['methods'][name]
                require(np.array_equal(value,finite_vector(actual['normalized_scores'],4,name)),'Stored final ranking score differs: '+name)
                pred=int(value.argmax());require(actual['prediction']==pred and actual['correct']==(pred==m['gold']),'Final ranking argmax differs')
            compact[qid]=dict(gold=m['gold'],methods=row['methods'],main_origin=row['main_origin'],lengths=m['choice_char_lengths'],
                counts=row['choice_token_counts'],prior_counts=row['prior_token_counts'])
        require(set(compact)==set(manifest),'Scored ID coverage differs')
        equal_fields(complete,dict(old_main_reuse_coverage=old_count,fresh_main_coverage=n-old_count,old_prior_cache_choice_hits=old_prior_hits,
            independent_prior_reference_coverage=0,actual_shared_depth=6,weights_modified=False,no_runtime_speed_claim=True),'Completion coverage')
        require(complete['logical_depths']==dict(context=5,prior=4) and complete['vocabulary_head_depths']==dict(context=[1,4,5],prior=[1,4]),'Logical readout dependencies differ')
        self.replay(task,complete,manifest,old_main)
        result=self.csv_and_stats(task,complete,manifest,compact)
        self.new_used.update(new_used)
        result.update(n=n,old_main_reuse_coverage=old_count,fresh_main_coverage=n-old_count,old_prior_cache_choice_hits=old_prior_hits,
            unique_new_prior_entries_used=len(new_used),context_vs_prior_target_boundary_changes=boundary_changes,
            independent_baseline_replay_receipt_coverage=complete['independent_baseline_parity_coverage'],independent_prior_reference_coverage=0,
            complete_sha256=self.hashes[complete_path],score_sha256=complete['score_file_sha256'])
        return result

    def old_main(self,task,spec,manifest,prov):
        if not spec.get('old_main_path'):
            require(prov['old_main_authority'] is None,'Unexpected old main authority');return {}
        self.bind(spec['old_main_path'],spec['old_main_sha256']);self.bind(spec['old_protocol_path'],spec['old_protocol_sha256'])
        protocol=read(spec['old_protocol_path']);old=read('FINAL_METHOD_LOCK.json');old_sha=self.bind('FINAL_METHOD_LOCK.json')
        worker_sha=self.bind('scripts/slcd_external_worker.py',protocol['worker_source_sha256'])
        require(protocol['manifest_sha256']==spec['manifest_sha256'] and protocol['n']==SIZES[task]
                and protocol['num_fewshot']==0 and protocol['harness_version']=='0.4.9' and protocol['final_method_lock_sha256']==old_sha,'Old task authority differs')
        for name,h in old['inference_source_sha256'].items():self.bind('scripts/'+name,h)
        for name,h in old['tokenizer_and_model_metadata_sha256'].items():self.bind('models/Ouro-1.4B/'+name,h)
        require(old['model_safetensors_sha256']==self.final['model_identity']['safetensors_sha256'],'Old model differs')
        require(all(importlib.metadata.version(k)==v for k,v in old['environment'].items()),'Old environment differs')
        for key,current in [('old_worker_source_sha256',worker_sha),('old_first_final_lock_sha256',old_sha)]:
            expected=spec.get(key,self.lock.get(key))
            if expected is not None:require(expected==current,'Optional old authority hash differs')
        equal_fields(prov['old_main_authority'],dict(file=spec['old_main_path'],sha256=spec['old_main_sha256'],protocol_sha256=spec['old_protocol_sha256'],
            old_worker_source_sha256=worker_sha,old_final_lock_sha256=old_sha),'Old main provenance')
        values=read_index(spec['old_main_path']);require(set(values)<=set(manifest),'Old main IDs outside manifest')
        for qid,row in values.items():
            m=manifest[qid]
            for key in ('gold','prompt','prompt_sha256','request_pairs','choice_char_lengths'):require(row[key]==m[key],'Old source-bound input differs')
            require(row['protocol_sha256']==spec['old_protocol_sha256'] and row['primary']==old['primary'],'Old row source binding differs')
            if 'choice_token_counts' in row:require(row['choice_token_counts']==[len(self.pair(*p)[1]) for p in m['request_pairs']],'Old recorded token counts differ')
            check_readouts(row['original_methods'],m['choice_char_lengths'],m['gold'])
        return values

    def replay(self,task,complete,manifest,old_main):
        relative='results/slcd_r6_'+task+'_BASELINE_REPLAY.json'
        self.bind(relative,complete['baseline_replay_receipt_sha256']);r=read(relative)
        ids=[qid for qid in manifest if qid in old_main]
        expected=[] if not ids else list(dict.fromkeys([ids[0],ids[len(ids)//2],ids[-1]]))
        require(r['status']=='passed' and r['ids']==expected and r['n']==len(expected)==complete['independent_baseline_parity_coverage']
                and r['eight_readouts_exact'] is True and r['max_sum_score_error']==0 and r['signature']==complete['signature']
                and r['old_main_reuse_coverage']==len(old_main),'Baseline replay receipt differs')

    def csv_and_stats(self,task,complete,manifest,compact):
        stem='results/slcd_r6_'+task
        required={stem+s for s in ('_results.csv','_paired_comparisons.csv','_paired_transitions.csv','_full_scores.csv')}
        require(set(complete['artifact_sha256'])==required,'Missing/unexpected CSV artifact')
        for p,h in complete['artifact_sha256'].items():self.bind(p,h)
        rng=np.random.default_rng(2026);comparisons=[]
        for name in METHODS:
            good=[compact[q]['methods'][name]['prediction']==compact[q]['gold'] for q in manifest]
            for baseline in ('fixed','A1_g1p0'):
                old=[compact[q]['methods'][baseline]['prediction']==compact[q]['gold'] for q in manifest]
                comparisons.append(dict(method=name,baseline=baseline,**statistics(good,old,rng)))
        results=[r for r in comparisons if r['baseline']=='fixed']
        for key,expected in [('results',results),('paired_comparisons',comparisons)]:
            require(len(complete[key])==len(expected),'Summary count differs')
            for actual,wanted in zip(complete[key],expected):equal_fields(actual,wanted,'Completion '+key)
        for suffix,expected in [('_results.csv',results),('_paired_comparisons.csv',comparisons)]:
            with path(stem+suffix).open(encoding='utf-8-sig',newline='') as stream:
                values=list(csv.DictReader(stream))
            require(len(values)==len(expected),'Summary CSV row count differs')
            for actual,wanted in zip(values,expected):equal_fields(actual,wanted,'Summary CSV',True)
        seen=set()
        with path(stem+'_paired_transitions.csv').open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                qid=row['id'];name=row['method'];key=(qid,name)
                require(qid in compact and name in METHODS and key not in seen,'Transition CSV ID/method duplicate');seen.add(key)
                r=compact[qid];m=r['methods'];pred=m[name]['prediction'];fixed=m['fixed']['prediction'];a1=m['A1_g1p0']['prediction']
                equal_fields(row,dict(gold=r['gold'],prediction=pred,correct=pred==r['gold'],fixed_prediction=fixed,
                    incumbent_prediction=a1,base_correct=fixed==r['gold'],incumbent_correct=a1==r['gold']),'Transition CSV',True)
        require(len(seen)==len(manifest)*5,'Incomplete transition CSV')
        transition_count=len(seen);seen=set()
        with path(stem+'_full_scores.csv').open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                qid=row['id'];name=row['method'];choice=int(row['choice']);key=(qid,name,choice)
                require(qid in compact and name in METHODS and 0<=choice<4 and key not in seen,'Full-score CSV ID/choice duplicate');seen.add(key)
                r=compact[qid];m=r['methods'][name]
                equal_fields(row,dict(normalized_score=m['normalized_scores'][choice],gold=r['gold'],prediction=m['prediction'],correct=m['correct'],
                    main_origin=r['main_origin'],choice_char_length=r['lengths'][choice],contextual_target_tokens=r['counts'][choice],
                    prior_target_tokens=r['prior_counts'][choice]),'Full-score CSV',True)
        require(len(seen)==len(manifest)*20,'Incomplete full-score CSV')
        return dict(status='passed',results=results,paired_comparisons=comparisons,
            transition_rows=transition_count,full_score_rows=len(seen),independent_formula_predictions_checked=len(manifest))


def self_test():
    """No artifact reads and no model/tokenizer imports; exercise tamper rejection."""
    from slcd_r6_scores import selected_score
    def readouts(offset=0.):
        result={}
        for i,name in enumerate(sorted(BASE)):
            raw=np.array([-2.,-3.,-4.,-5.])+offset+i*.125;norm=raw/np.array([1,2,3,4]);p=int(norm.argmax())
            result[name]=dict(sum_scores=raw.tolist(),normalized_scores=norm.tolist(),prediction=p,correct=p==2)
        return result
    original=readouts();prior=readouts(.5);check_readouts(original,[1,2,3,4],2)
    values=ranking_scores(original,prior,selected_score);require(set(values)==set(METHODS),'Synthetic method set')
    bad=json.loads(json.dumps(original));bad['native_k4']['normalized_scores'][0]+=.1
    try:check_readouts(bad,[1,2,3,4],2)
    except ValueError:pass
    else:raise AssertionError('Corrupted normalized score accepted')
    s=statistics([True,False,True,False],[False,True,False,False],np.random.default_rng(2026))
    require(s['corrections']==2 and s['regressions']==1 and s['net_correct']==1 and s['exact_mcnemar_p']==1.,'Synthetic paired statistics')
    equal_fields({'correct':'True','n':'4','accuracy':'0.5'},dict(correct=True,n=4,accuracy=.5),'Synthetic CSV',True)
    try:equal_fields({'n':'3'},dict(n=4),'Synthetic tamper',True)
    except ValueError:pass
    else:raise AssertionError('Corrupted CSV accepted')
    audit=Audit();audit.pair=lambda context,continuation:([1,2],[3,4])
    entry=dict(signature='synthetic',prefix='Answer:',continuation=' example',context_ids=[1,2],target_ids=[3,4],sum_scores={k:-1. for k in BASE})
    entry['key']=digest(dict(global_signature='synthetic',prefix='Answer:',continuation=' example',context_ids=[1,2],target_ids=[3,4]))
    audit.cache_entry(entry,'synthetic')
    for mode in ('key','boundary'):
        corrupt=json.loads(json.dumps(entry))
        if mode=='key':corrupt['key']='wrong'
        else:
            corrupt['target_ids']=[3]
            corrupt['key']=digest(dict(global_signature='synthetic',prefix='Answer:',continuation=' example',context_ids=[1,2],target_ids=[3]))
        try:audit.cache_entry(corrupt,'synthetic')
        except ValueError:pass
        else:raise AssertionError('Corrupted prior cache accepted: '+mode)
    print(json.dumps(dict(status='passed',scope='synthetic only',tests=['independent ranking','normalization tamper rejection','paired counts/McNemar/bootstrap','typed CSV and tamper rejection','cache key and target-boundary tamper rejection']),indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--task',choices=['all']+list(SIZES),default='all');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    audit=Audit();audit.initialize();tasks=list(SIZES) if args.task=='all' else [args.task]
    result={task:audit.task(task) for task in tasks}
    if args.task=='all':require(audit.new_used==set(audit.new_cache),'Unreferenced new external cache entries after both tasks completed')
    # Freeze evidence against mutation during this independent CPU audit.
    for relative,expected in audit.hashes.items():require(file_sha(read_path(relative))==expected,'Artifact changed during audit: '+relative)
    for relative,expected in audit.dependency_hashes.items():require(file_sha(audit.harness_distribution.locate_file(relative))==expected,'Installed task source changed during audit')
    report=dict(status='passed',scope='combined' if args.task=='all' else args.task,primary=PRIMARY,tasks=result,
        external_lock_sha256=audit.lock_sha,final_lock_sha256=audit.final_sha,artifact_sha256=audit.hashes,
        audit_source_sha256=file_sha(Path(__file__)),package_versions=audit.versions,encode_pair_source_sha256=audit.pair_source_sha,
        installed_task_source_sha256=audit.dependency_hashes,
        token_pairs_reencoded=audit.token_pairs_checked,prior_cache_entries_checked=dict(old_arc=len(audit.old_cache),new_external=len(audit.new_cache)),
        checks=dict(complete_before_any_outcome_read=True,source_and_model_file_hashes=True,manifest_and_candidate_order=True,
            native_readout_sums_normalization=True,five_final_rankings=True,selected_formula_independent_crosscheck=True,
            prior_entry_digests_and_raw_sums=True,all_target_ids_reencoded=True,both_contexts_complete_no_truncation=True,
            all_four_csv_artifacts=True,paired_statistics_recomputed=True,chain_receipts=True),
        limits=['No GPU or model forward execution; numeric likelihoods authenticated and recombined, not independently regenerated.',
            'Baseline replay coverage is checked from the worker receipt; this CPU audit does not itself rerun those trajectories.',
            'No independent historical auxiliary-prior reference exists.',
            'Contextual prefix token arrays were not persisted; exact prompt bytes/source/tokenizer are bound and tokenization is recomputed.',
            'Official raw dataset/task-source files and manifest are hash-bound; this audit does not regenerate the task manifest through ConfigurableTask.',
            'P-values are raw pairwise McNemar values; bootstrap intervals are marginal, not multiplicity-adjusted.',
            'No new method selection, causal mechanism proof, publication-timing proof, or runtime speed claim.'])
    destination=path('results/slcd_R6_EXTERNAL_ACCEPTANCE.json' if args.task=='all' else f'results/slcd_R6_{args.task}_ACCEPTANCE.json')
    require(destination.parent.is_dir(),'Output parent missing')
    temporary=destination.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');temporary.replace(destination)
    print(json.dumps(dict(status='passed',scope=report['scope'],output=str(destination),correct={t:next(x['correct'] for x in result[t]['results'] if x['method']==PRIMARY) for t in tasks}),indent=2))


if __name__=='__main__':main()
