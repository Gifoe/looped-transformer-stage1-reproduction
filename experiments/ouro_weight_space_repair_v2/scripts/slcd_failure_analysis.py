"""Train-only observable feature audit; never a fitted decision model."""
from slcd_core import *


def features(row):
    m=row['methods'];p={k:int(m['native_k'+str(k)].argmax()) for k in range(1,7)}
    fixed=m['loopcd_k4_fixed'];q=int(fixed.argmax());native=m['native_k4'];old=p[4]
    return dict(native_margin=top_margin(native),fixed_margin=top_margin(fixed),
        native_k3_k4_agree=p[3]==p[4],native_k4_k5_agree=p[4]==p[5],
        native_k3_k4_k5_agree=p[3]==p[4]==p[5],native_k4_k5_k6_agree=p[4]==p[5]==p[6],
        fixed_k5_agree=q==p[5],fixed_k3_agree=q==p[3],
        late_opposition=float((native-m['native_k5'])[q]-(native-m['native_k5'])[old]),
        contrast_pair_displacement=float((fixed-native)[q]-(fixed-native)[old]),
        candidate_count=len(native))


def main():
    report=[];cases=[]
    for split in ['discovery','selection']:
        data=load_development(split)
        for row in data:
            n=row['methods']['native_k4'];c=row['methods']['loopcd_k4_fixed'];y=row['gold']
            old=int(n.argmax());new=int(c.argmax())
            if old==new:continue
            typ='rescue' if new==y else 'damage' if old==y else 'wrong_to_wrong'
            cases.append(dict(split=split,id=row['id'],transition=typ,native_prediction=old,
                fixed_prediction=new,gold=y,**features(row)))
        for condition in ['all_changed','native_k3_k4_agree','native_k4_k5_agree','native_k3_k4_k5_agree',
                'native_k4_k5_k6_agree','fixed_k5_agree','fixed_k3_agree','native_margin_ge_0p025',
                'native_margin_ge_0p05','fixed_margin_le_0p025','late_opposition_nonnegative']:
            selected=[]
            for c in cases:
                if c['split']!=split:continue
                if condition=='all_changed':flag=True
                elif condition=='native_margin_ge_0p025':flag=c['native_margin']>=.025
                elif condition=='native_margin_ge_0p05':flag=c['native_margin']>=.05
                elif condition=='fixed_margin_le_0p025':flag=c['fixed_margin']<=.025
                elif condition=='late_opposition_nonnegative':flag=c['late_opposition']>=0
                else:flag=c[condition]
                if flag:selected.append(c)
            counts={name:sum(c['transition']==name for c in selected) for name in ['rescue','damage','wrong_to_wrong']}
            report.append(dict(split=split,condition=condition,n_changed_cases=len(selected),**counts,
                net_recovered_if_revert_to_native=counts['damage']-counts['rescue'],
                mean_native_margin=float(np.mean([c['native_margin'] for c in selected])) if selected else None))
    writecsv('results/slcd_failure_feature_audit.csv',report)
    writecsv('results/slcd_development_changed_cases.csv',cases)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
