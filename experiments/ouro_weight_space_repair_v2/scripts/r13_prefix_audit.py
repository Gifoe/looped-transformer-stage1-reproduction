"""Development-only complete-prefix score accounting, never a new decoder.

All scalar positions were produced by actual R7 vocabulary-normalized forwards.
Target/partition components have FP32 rounding residuals, explicitly retained.
Counterfactual component removal is algebra, not a valid normalized LM or a
hidden-state causal intervention. No heldout reader, GPU, fitting or selection.
"""
import csv,json,math
from collections import defaultdict
import numpy as np
from r7_core import ROOT,read,sha,jsonlines

PROTOCOL='configs/r13_PREFIX_AUDIT_PROTOCOL.json'


def csvwrite(path,data):
    with (ROOT/path).open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)


def main():
    spec=read(PROTOCOL);assert sha('scripts/r13_prefix_audit.py')==spec['source_sha256']
    assert spec['GPU_seconds']==0 and not spec['heldout_access_authorized']
    for p,h in spec['input_sha256'].items():assert sha(p)==h
    allrows=[]
    for split,expected in [('discovery',384),('selection',192)]:
        path=f'results/r7_{split}_development.jsonl';receipt=read(path.replace('.jsonl','_COMPLETE.json'))
        assert receipt['status']=='complete' and receipt['n']==expected and sha(path)==receipt['score_file_sha256']
        data=list(jsonlines(path));assert len(data)==len({r['id'] for r in data})==expected
        for row in data:
            assert row['signature']==receipt['signature'];allrows.append((split,row))
    assert len(allrows)==576 and sum(r['correct']['M0_R6'] for _,r in allrows)==360
    detailed=[];perchoice=[];groups=defaultdict(list);max_scalar_error=0.;max_score_error=0.;max_margin_error=0.;max_bound_ratio=0.
    unit=np.finfo(np.float32).eps/2
    for split,row in allrows:
        lengths=np.asarray(row['choice_char_lengths']);a=np.asarray(row['baseline_scores']['A1']);r6=np.asarray(row['baseline_scores']['M0_R6']);gold=row['gold']
        late=np.asarray(row['original_methods']['native_k4']['normalized_scores'])-np.asarray(row['original_methods']['native_k5']['normalized_scores'])
        original_sums=np.asarray(row['original_methods']['loopcd_k4_fixed']['sum_scores']);n=len(a)
        assert len(row['token_diagnostics'])==n and len(row['token_counts'])==n
        for name in spec['methods']:
            assert not row['alignment_fallback_by_method'][name]
            new=np.asarray(row['method_scores'][name]);target=[];partition=[];first=[];later=[];bounds=[];score_residuals=[]
            for i,d in enumerate(row['token_diagnostics']):
                d=d[name];count=row['token_counts'][i]
                z=np.asarray(d['target_logit_shift']);c=np.asarray(d['log_normalizer_change']);lp=np.asarray(d['target_logprob_change'])
                assert len(z)==len(c)==len(lp)==len(d['alpha'])==count and count>0
                assert np.isfinite([z,c,lp,d['alpha']]).all()
                identity=z-c-lp;scalar_err=float(np.max(np.abs(identity)));max_scalar_error=max(max_scalar_error,scalar_err)
                # Conservative rounding bound for stored FP32 targetshift-delta.
                scalar_bound=2*unit*(np.abs(z)+np.abs(lp))+np.finfo(np.float64).eps
                assert np.all(np.abs(identity)<=scalar_bound)
                change=math.fsum(lp)/lengths[i];reconstructed=original_sums[i]/lengths[i]+late[i]+change
                residual=float(new[i]-reconstructed);max_score_error=max(max_score_error,abs(residual))
                # Positive sums of |log probabilities| can be bounded from their
                # FP32 reductions. This is deliberately conservative, not tuned.
                gamma=count*unit/(1-count*unit);new_sum=(new[i]-late[i])*lengths[i]
                bound=(gamma+unit)/(1-gamma)*(abs(new_sum)+abs(original_sums[i]))/lengths[i]+64*np.finfo(np.float64).eps*max(abs(new[i]),1.)
                assert abs(residual)<=bound
                t=math.fsum(z)/lengths[i];p=-math.fsum(c)/lengths[i]
                target.append(t);partition.append(p);first.append(float(lp[0]/lengths[i]));later.append(math.fsum(lp[1:])/lengths[i]);bounds.append(bound+math.fsum(abs(identity))/lengths[i]);score_residuals.append(residual)
                perchoice.append(dict(split=split,id=row['id'],method=name,choice=i,tokens=count,characters=int(lengths[i]),target_component=t,partition_component=p,
                    first_position_LP_change=first[-1],later_positions_LP_change=later[-1],actual_score_change=float(new[i]-a[i]),
                    scalar_identity_max_error=scalar_err,FP32_reduction_residual=residual,conservative_score_error_bound=bound,
                    alpha_first=d['alpha'][0],alpha_mean=float(np.mean(d['alpha'])),alpha_later_mean=float(np.mean(d['alpha'][1:])) if count>1 else None))
            assert row['predictions'][name]==int(new.argmax()) and row['correct'][name]==(int(new.argmax())==gold)
            rivals={'pre_A1_rival':max((j for j in range(n) if j!=gold),key=lambda j:a[j]),
                    'post_repair_rival':max((j for j in range(n) if j!=gold),key=lambda j:new[j])}
            for frame,rival in rivals.items():
                T=target[gold]-target[rival];P=partition[gold]-partition[rival];F=first[gold]-first[rival];L=later[gold]-later[rival]
                before=float(a[gold]-a[rival]);after=float(new[gold]-new[rival]);residual=after-before-T-P;bound=bounds[gold]+bounds[rival]
                max_margin_error=max(max_margin_error,abs(residual));max_bound_ratio=max(max_bound_ratio,abs(residual)/bound if bound else 0.);assert abs(residual)<=bound
                undo=float((a[gold]-a[rival])-(r6[gold]-r6[rival]));r6before=float(r6[gold]-r6[rival]);assert abs(after-r6before-(undo+T+P+residual))<1e-12
                damage=row['correct']['A1'] and not row['correct'][name];rescue=not row['correct']['A1'] and row['correct'][name]
                fixed_cross=before>0 and after<0
                detail=dict(split=split,id=row['id'],method=name,rival_frame=frame,gold=gold,rival=rival,A1_correct=row['correct']['A1'],R6_correct=row['correct']['M0_R6'],new_correct=row['correct'][name],
                    A1_margin=before,new_margin=after,target_component=T,partition_component=P,FP32_margin_residual=residual,conservative_error_bound=bound,
                    first_position_LP_component=F,later_positions_LP_component=L,R6_margin=r6before,undo_R6_component=undo,
                    target_nonharmful_partition_negative=bool(T>=0 and P<0),partition_necessary_in_algebra=bool(fixed_cross and before+T>=0),
                    target_necessary_in_algebra=bool(fixed_cross and before+P>=0),first_necessary_in_algebra=bool(fixed_cross and before+L>=0),
                    later_necessary_in_algebra=bool(fixed_cross and before+F>=0),old_rival_crossed=bool(fixed_cross),
                    R6_unique_rescue=row['correct']['M0_R6'] and not row['correct']['fixed'])
                detailed.append(detail)
                transition='Correct->Wrong' if damage else 'Wrong->Correct' if rescue else 'Correct->Correct' if row['correct']['A1'] else 'Wrong->Wrong'
                groups[(name,frame,'A1',transition)].append(detail)
                r6transition=('Correct' if row['correct']['M0_R6'] else 'Wrong')+'->'+('Correct' if row['correct'][name] else 'Wrong')
                groups[(name,frame,'R6',r6transition)].append(detail)
    summary=[]
    for (name,frame,reference,transition),data in sorted(groups.items()):
        summary.append(dict(method=name,rival_frame=frame,reference=reference,transition=transition,n=len(data),
            target_nonharmful_partition_negative=sum(r['target_nonharmful_partition_negative'] for r in data),
            partition_necessary_in_algebra=sum(r['partition_necessary_in_algebra'] for r in data),target_necessary_in_algebra=sum(r['target_necessary_in_algebra'] for r in data),
            first_necessary_in_algebra=sum(r['first_necessary_in_algebra'] for r in data),later_necessary_in_algebra=sum(r['later_necessary_in_algebra'] for r in data),
            old_rival_crossed=sum(r['old_rival_crossed'] for r in data),target_component_mean=float(np.mean([r['target_component'] for r in data])),
            partition_component_mean=float(np.mean([r['partition_component'] for r in data])),undo_R6_component_mean=float(np.mean([r['undo_R6_component'] for r in data]))))
    csvwrite('results/r13_prefix_margin_decomposition.csv',detailed);csvwrite('results/r13_prefix_choice_components.csv',perchoice);csvwrite('results/r13_prefix_summary.csv',summary)
    result=dict(status='complete_development_prefix_accounting',n=576,methods=spec['methods'],CPU_only=True,GPU_seconds=0,heldout_accessed=False,protocol_sha256=sha(PROTOCOL),
        candidate_records=len(perchoice),all_prediction_positions=sum(r['tokens'] for r in perchoice),rival_margin_rows=len(detailed),max_scalar_identity_error=max_scalar_error,max_score_reconstruction_residual=max_score_error,
        max_margin_decomposition_residual=max_margin_error,max_margin_error_to_conservative_bound=max_bound_ratio,
        summary=summary,outputs={p:sha(p) for p in ['results/r13_prefix_margin_decomposition.csv','results/r13_prefix_choice_components.csv','results/r13_prefix_summary.csv']},
        interpretation='Posthoc complete development accounting only. Equal-weight question rows, no token independence claim. Residuals explicitly FP32. Removing a log-partition term does not define a valid normalized LM or establish hidden-state causation. No candidate or coefficient was selected.')
    with (ROOT/'results/r13_PREFIX_AUDIT_COMPLETE.json').open('x',encoding='utf8') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='summary'}),flush=True)


if __name__=='__main__':main()
